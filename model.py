import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import sobel
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


# ------------------------------------------------------------
# Função de distância geográfica (Haversine)
# ------------------------------------------------------------
def haversine(lon1, lat1, lon2, lat2):
    """Distância em metros entre dois pontos (lon/lat)"""
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ------------------------------------------------------------
# Classe principal do modelo
# ------------------------------------------------------------
class FloodModel:
    """
    Modelo robusto de previsão de risco de alagamento
    baseado em dados topográficos (SRTM), espectrais (Sentinel)
    e eventos reais (CSV).
    """

    def __init__(self, slope_tif, rough_tif, aspect_tif, sentinel_tif, occ_csv="Dados/csv/alagamentos_uberlandia.csv", weights=None):
        self.slope_tif = slope_tif
        self.rough_tif = rough_tif
        self.aspect_tif = aspect_tif
        self.sentinel_tif = sentinel_tif
        self.occ_csv = occ_csv

        # --- Raster base (declividade) ---
        with rasterio.open(slope_tif) as ref:
            self.profile = ref.profile
            self.ref_transform = ref.transform
            self.ref_crs = ref.crs
            self.ref_height = ref.height
            self.ref_width = ref.width
            self.slope = ref.read(1).astype("float32")

        # --- Alinhamento dos demais ---
        self.rough = self._align(rough_tif)
        self.aspect = self._align(aspect_tif)
        self.R, self.G, self.B = self._align_rgb(sentinel_tif)

        # --- Derivados topográficos e espectrais ---
        self.flatness = 1 - self._normalize(self.slope, 0, 30)
        self.smoothness = 1 - self._normalize(self.rough)
        self.ndwi = self._calc_ndwi()
        self.ndbi = self._calc_ndbi()
        self.twi = self._calc_twi()
        self.flow_acc = self._calc_flow_proxy()
        self.impervious = self._impervious()

        # --- Pesos calibrados empiricamente ---
        self.weights = weights or {
            "twi": 0.30,
            "flatness": 0.20,
            "smoothness": 0.05,
            "impervious": 0.30,
            "flow_acc": 0.10,
            "ndwi": 0.05
        }

        # --- Índice base de suscetibilidade ---
        self.base_score = self._calc_base_score()

        # --- Carregar CSV de ocorrências ---
        try:
            self.df_occ = pd.read_csv(self.occ_csv)
            print(f"[INFO] {len(self.df_occ)} ocorrências carregadas do CSV.")
        except Exception as e:
            print(f"[AVISO] Falha ao carregar CSV de ocorrências: {e}")
            self.df_occ = None

    # ------------------------------------------------------------
    # Funções auxiliares de alinhamento e normalização
    # ------------------------------------------------------------
    def _align(self, src_path, resampling=Resampling.bilinear):
        with rasterio.open(src_path) as src:
            arr = np.zeros((self.ref_height, self.ref_width), dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.ref_transform,
                dst_crs=self.ref_crs,
                resampling=resampling
            )
        return arr

    def _align_rgb(self, src_path):
        with rasterio.open(src_path) as s:
            out = []
            for b in [1, 2, 3]:
                arr = np.zeros((self.ref_height, self.ref_width), dtype="float32")
                reproject(
                    source=rasterio.band(s, b),
                    destination=arr,
                    src_transform=s.transform,
                    src_crs=s.crs,
                    dst_transform=self.ref_transform,
                    dst_crs=self.ref_crs,
                    resampling=Resampling.bilinear
                )
                out.append(arr)
        return out

    def _normalize(self, arr, low=None, high=None):
        mask = ~np.isfinite(arr)
        if low is None:
            low = np.nanpercentile(arr[~mask], 2)
        if high is None:
            high = np.nanpercentile(arr[~mask], 98)
        arr = np.clip(arr, low, high)
        out = (arr - low) / (high - low + 1e-6)
        out[mask] = np.nan
        return out

    # ------------------------------------------------------------
    # Indicadores derivados
    # ------------------------------------------------------------
    def _calc_ndwi(self):
        ndwi = (self.G - self.R) / (self.G + self.R + 1e-6)
        return self._normalize(ndwi, -0.5, 0.5)

    def _calc_ndbi(self):
        ndbi = (self.R - self.G) / (self.R + self.G + 1e-6)
        return self._normalize(ndbi, -0.5, 0.5)

    def _impervious(self):
        brightness = self._normalize((self.R + self.G + self.B) / 3.0)
        imperv = self._normalize(brightness * (1 - self.ndwi))
        return imperv

    def _calc_twi(self):
        slope_rad = np.deg2rad(self.slope)
        twi = np.log(1 + (1 / (np.tan(slope_rad) + 1e-6)))
        return self._normalize(twi)

    def _calc_flow_proxy(self):
        dzdx = sobel(self.slope, axis=1)
        dzdy = sobel(self.slope, axis=0)
        grad_mag = np.hypot(dzdx, dzdy)
        flow_acc = self._normalize(1 - grad_mag / (grad_mag.max() + 1e-6))
        return flow_acc

    # ------------------------------------------------------------
    # Índice de suscetibilidade base
    # ------------------------------------------------------------
    def _calc_base_score(self):
        w = self.weights
        base = (
            w["twi"] * self.twi +
            w["flatness"] * self.flatness +
            w["smoothness"] * self.smoothness +
            w["impervious"] * self.impervious +
            w["flow_acc"] * self.flow_acc +
            w["ndwi"] * self.ndwi
        )
        return self._normalize(base)

    # ------------------------------------------------------------
    # Ajuste de peso por proximidade (CSV)
    # ------------------------------------------------------------
    def proximity_weight(self, lon, lat, raio_m=1200):
        """Aumenta o risco em regiões próximas a eventos históricos (versão mais moderada)"""
        if self.df_occ is None or self.df_occ.empty:
            return 1.0
        dists = self.df_occ.apply(lambda r: haversine(lon, lat, r["lon"], r["lat"]), axis=1)
        near = self.df_occ[dists <= raio_m]
        if near.empty:
            return 1.0
        # Peso mais moderado para evitar saturação prematura
        peso = 1 + (len(near) / 8) + ((near["chuva_mm"].mean(skipna=True) or 0) / 200)
        return min(peso, 1.8)  # limite reduzido de 2.5 para 1.8

    # ------------------------------------------------------------
    # Modelo de risco com chuva (baseado em intensidade mm/min)
    # ------------------------------------------------------------
    def risk_with_rain(self, base, mm, freq_min, lon=None, lat=None):
        """
        Probabilidade de alagamento melhorada com:
        - base: suscetibilidade topográfica (0–1)
        - mm: volume total de chuva (mm)
        - freq_min: duração da chuva (minutos)
        - ajuste por proximidade (CSV)
        """
        # intensidade da chuva (mm/min)
        intensity = mm / max(freq_min, 1)
        
        # Escala de intensidade mais sensível e realista
        # 0.5 mm/min (30 mm/h) = baixo risco
        # 1.0 mm/min (60 mm/h) = médio risco  
        # 2.0 mm/min (120 mm/h) = alto risco
        int_scale = np.tanh(intensity / 1.5)  # mais suave, evita saturação
        
        # Volume acumulado com escala mais realista
        mm_scale = np.tanh(mm / 60.0)  # 60mm como referência média
        
        # Peso base reduzido para dar mais importância à chuva
        base_weight = 0.4 + (0.3 * base)  # varia de 0.4 a 0.7
        
        # Combinação rebalanceada - pesos menores para evitar saturação
        lin = (base * base_weight) + (mm_scale * 0.8) + (int_scale * 1.2) - 1.0
        
        # Função logística menos agressiva
        prob = 1 / (1 + np.exp(-2.5 * lin))
        
        # Ajuste por proximidade mais moderado
        if lon is not None and lat is not None:
            prox_weight = self.proximity_weight(lon, lat)
            # Reduzir impacto da proximidade para evitar saturação
            prox_factor = 1.0 + (prox_weight - 1.0) * 0.5  # metade do impacto original
            prob *= prox_factor

        return float(np.clip(prob, 0, 1))

    # ------------------------------------------------------------
    # Inferência em ponto
    # ------------------------------------------------------------
    def sample_point(self, x, y, mm, freq_min, modo="geo"):
        if modo == "geo":
            with rasterio.open(self.slope_tif) as src:
                row, col = src.index(x, y)
        elif modo == "cart":
            col, row = int(x), int(y)
        else:
            raise ValueError("modo deve ser 'geo' ou 'cart'")

        if row < 0 or row >= self.ref_height or col < 0 or col >= self.ref_width:
            raise IndexError("Coordenadas fora dos limites")

        base_val = float(self.base_score[row, col])
        prob = self.risk_with_rain(base_val, mm, freq_min, lon=x, lat=y)
        return {"x": x, "y": y, "risco_base": base_val, "probabilidade": prob}

    def radius_influence(self, x, y, mm, freq_min, radius_m=500):
        res_deg = self.profile["transform"][0]
        deg_per_m = 1 / 111320
        pix_radius = int(radius_m * deg_per_m / res_deg)
        with rasterio.open(self.slope_tif) as src:
            row, col = src.index(x, y)
        r1, r2 = max(0, row - pix_radius), min(self.ref_height, row + pix_radius)
        c1, c2 = max(0, col - pix_radius), min(self.ref_width, col + pix_radius)
        area = self.base_score[r1:r2, c1:c2]
        mean_base = np.nanmean(area)
        prob = self.risk_with_rain(mean_base, mm, freq_min, lon=x, lat=y)
        return {"raio_m": radius_m, "prob_média": prob}

    def get_bounds(self):
        with rasterio.open(self.slope_tif) as src:
            b = src.bounds
            return {
                "oeste": b.left,
                "leste": b.right,
                "sul": b.bottom,
                "norte": b.top,
                "largura_px": self.ref_width,
                "altura_px": self.ref_height
            }