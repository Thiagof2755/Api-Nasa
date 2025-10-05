# Flood Risk API Documentation

## Overview
This FastAPI service estimates flood susceptibility for locations in Uberlandia by combining topographic rasters from NASA SRTM, Sentinel-2 optical imagery, and a catalog of historical flood events. The service exposes endpoints for single-point and batch analysis, returning the underlying susceptibility score and the probability of flooding once rainfall intensity is provided.

## Data Inputs and Pre-processing
- **Slope (`viz.SRTMGL1_slope.tif`)**: serves as the reference grid; the API extracts CRS, transform, and raster dimensions here.
- **Roughness (`viz.SRTMGL1_roughness.tif`)** and **Aspect (`viz.SRTMGL1_aspect.tif`)**: reprojected and resampled to the slope grid.
- **Sentinel-2 Atmospheric Penetration composite**: bands 1, 2, and 3 (R, G, B) reprojected to the slope grid.
- **Historical flood CSV (`csv/alagamentos_uberlandia.csv`)** (optional): used to boost risk near documented occurrences.
- All rasters are normalized via percentile clipping (2nd to 98th percentile) to reduce the influence of outliers or missing pixels.

## Derived Indicators
The model synthesizes the aligned rasters into physically meaningful indices:
- **Flatness**: `1 - normalize(slope, 0, 30)`; flatter terrain raises susceptibility.
- **Smoothness**: `1 - normalize(roughness)`; smoother surfaces correlate with water retention.
- **NDWI**: `(Green - Red) / (Green + Red)` normalized in [-0.5, 0.5] to highlight surface water and moisture.
- **NDBI**: `(Red - Green) / (Red + Green)` normalized in [-0.5, 0.5]; helps separate built-up areas.
- **Imperviousness**: normalized brightness scaled by `(1 - NDWI)`; proxies impermeable surfaces.
- **TWI (Topographic Wetness Index)**: `log(1 + 1 / tan(slope))`, normalized; higher values indicate accumulation zones.
- **Flow Accumulation Proxy**: inverse gradient magnitude obtained with Sobel filters; high values simulate drainage convergence.

These features are combined with empirically tuned weights to form the **base susceptibility score**:
```
base = 0.30*TWI + 0.20*Flatness + 0.05*Smoothness +
        0.30*Impervious + 0.10*FlowAcc + 0.05*NDWI
score = normalize(base)
```

## Rainfall-driven Risk Model
Given rainfall totals and duration, the model maps susceptibility to a flood probability:
1. **Rain intensity**: `intensity = chuva_mm / max(freq_min, 1)`; scaled with `tanh(intensity / 1.5)`.
2. **Accumulated rain**: `tanh(chuva_mm / 60)` emphasizes totals around 60 mm as a high-risk reference.
3. **Base weight**: `0.4 + 0.3 * base_score` (0.4–0.7) retains terrain influence without overpowering rain.
4. **Linear combination**: `lin = base*base_weight + 0.8*mm_scale + 1.2*int_scale - 1.0`.
5. **Logistic mapping**: `probability = 1 / (1 + exp(-2.5 * lin))` clamps results to [0, 1].
6. **Historical events proximity**: Haversine distance checks the CSV for events within 1.2 km. The probability is multiplied by `1 + 0.5*(weight - 1)`, where `weight <= 1.8` increases with event density and rain recorded during those events.

## API Endpoints

### `GET /`
Returns API metadata and a quick summary of available endpoints.

### `POST /analisar`
- **Purpose**: compute flood probability for a single location.
- **Request body** (`application/json`):
  - `lon` (float): longitude in the raster CRS; defaults to WGS84 degrees when `modo="geo"`.
  - `lat` (float): latitude.
  - `chuva_mm` (float): rainfall depth accumulated for the event window (mm).
  - `freq_min` (int): duration of the rainfall window (minutes).
  - `modo` (string, optional): `"geo"` (default) to geocode lon/lat, or `"cart"` to supply raster column/row indices directly.
- **Processing steps**:
  1. Convert coordinates to raster indices (`rasterio.index` when `modo="geo"`).
  2. Read the cell-level base susceptibility score.
  3. Run the rainfall model (`risk_with_rain`) for the provided rain metrics.
  4. Compute a neighborhood probability by averaging the base score inside a 500 m radius buffer and re-evaluating the rainfall model.
- **Success response (HTTP 200)**:
```json
{
  "entrada": { ... original request ... },
  "probabilidade": 0.62,
  "risco_base": 0.55,
  "raio_influencia": {
    "raio_m": 500,
    "prob_media": 0.58
  },
  "status": "sucesso"
}
```
- **Error responses**:
  - Out-of-bounds coordinates return `status: "erro"` with `"Coordenadas fora da area de cobertura"`.
  - Any unexpected failure produces `"Erro interno"` with diagnostic details.

### `POST /analisar-batch`
- **Purpose**: evaluate multiple points in one request.
- **Request body**:
```json
{
  "pontos": [ { single-point payload }, ... ]
}
```
- **Processing steps** (per item): identical to `POST /analisar`.
- **Response fields**:
  - `total_pontos`: number of inputs received.
  - `sucessos`: count processed without errors.
  - `erros_count`: count of failures; details per item under `erros`.
  - `resultados`: array of success objects (`indice`, `entrada`, `probabilidade`, `risco_base`, `raio_influencia`).
  - `arrays`: parallel arrays extracted from `resultados` for downstream analytics (probabilities, base risks, radius probabilities).
  - `estatisticas`: min, max, and mean of the probability array when available.

## Supporting Methods (not exposed as endpoints)
- `FloodModel.radius_influence`: averages the base susceptibility inside a radius (default 500 m) before re-running the rainfall probability.
- `FloodModel.get_bounds`: reports raster bounds and size; you may expose it as a `GET /extensao` endpoint if needed.

## Usage Notes
- Coordinate mode must match the raster CRS (typically EPSG:4326 for the provided rasters). Invalid points trigger an error.
- Probabilities are calibrated for intense rain pulses (>=30 mm/h). Low intensity rain may still return non-zero probabilities if the base terrain risk is high.
- If the historical CSV is missing, the model falls back to pure raster-based estimation without proximity weighting.
- Extend the API by wiring `FloodModel.get_bounds()` to a `GET /extensao` route if you need to publish raster coverage.

## Sample `curl`
```bash
curl -X POST http://localhost:8000/analisar \
     -H "Content-Type: application/json" \
     -d '{
           "lon": -48.261,
           "lat": -18.914,
           "chuva_mm": 45.0,
           "freq_min": 60,
           "modo": "geo"
         }'
```

The response will include the computed probability (0-1), the underlying susceptibility score, and the neighborhood influence summary.
