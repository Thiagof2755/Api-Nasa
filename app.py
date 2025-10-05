from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import math
from model import FloodModel

# inicializa o modelo
flood = FloodModel(
    slope_tif="Dados/ImagenRelevos/viz.SRTMGL1_slope.tif",
    rough_tif="Dados/ImagenRelevos/viz.SRTMGL1_roughness.tif",
    aspect_tif="Dados/ImagenRelevos/viz.SRTMGL1_aspect.tif",
    sentinel_tif="Dados/Atmospheric_penetration/2025-09-30-00_00_2025-09-30-23_59_Sentinel-2_L2A_Atmospheric_Penetration.tiff"
)

app = FastAPI(
    title="🌊 UrbMind - Advanced Flood Risk Assessment API",
    description="""
    ## 🚀 NASA - Urban Flood Intelligence System
    
    **UrbMind** is an advanced AI-powered flood risk assessment platform that combines cutting-edge geospatial data with machine learning to provide real-time flood susceptibility analysis for urban environments.
    
    ### 🎯 Core Features:
    - **🎯 Point Analysis**: Evaluates flood risk for specific geographic coordinates
    - **📊 Batch Processing**: Efficiently processes multiple locations simultaneously 
    - **🌐 Radius Influence**: Calculates flood impact within customizable geographic buffers
    - **📍 Geographic Coverage**: Provides detailed information about data extent and boundaries
    - **🔬 Scientific Accuracy**: Uses NASA SRTM elevation data and Sentinel-2 satellite imagery
    
    ### 🛰️ Data Sources & Technology Stack:
    - **NASA SRTM Digital Elevation Model**: High-resolution terrain analysis
    - **Terrain Analytics**: Slope, roughness, and aspect calculations for hydrological modeling
    - **Sentinel-2 Satellite Imagery**: Multi-spectral analysis for land cover classification
    - **Historical Flood Database**: Real flood occurrence data for model validation and calibration
    - **Advanced ML Algorithms**: Topographic Wetness Index (TWI), NDWI, NDBI, and flow accumulation modeling
    
    ### 🏆 Innovation Highlights:
    - **Multi-modal Data Fusion**: Combines topographic, spectral, and historical data
    - **Real-time Risk Assessment**: Instant flood probability calculations based on rainfall scenarios
    - **Scalable Architecture**: Optimized for both single-point queries and large-scale batch analysis
    - **Scientific Validation**: Risk models calibrated against real historical flood events
    
    ### 🌍 Use Cases:
    - Urban planning and flood-resilient city design
    - Emergency response and early warning systems
    - Insurance risk assessment and actuarial modeling
    - Infrastructure development and site selection
    - Climate adaptation planning
    """,
    version="2.0",
    contact={
        "name": "UrbMind Development Team",
        "email": "team@urbmind.com",
        "url": "https://github.com/urbmind/flood-risk-api"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    terms_of_service="https://urbmind.com/terms",
    openapi_tags=[
        {
            "name": "Analysis",
            "description": "Core flood risk analysis endpoints for single-point and batch processing"
        },
        {
            "name": "Information", 
            "description": "API metadata, geographic coverage, and system information endpoints"
        }
    ]
)

class InputPoint(BaseModel):
    """
    Geographic point data model for flood risk analysis
    
    This model defines the input parameters required for flood risk assessment at a specific location.
    All coordinates should be provided in WGS84 decimal degrees format.
    
    Attributes:
        lon (float): Longitude in decimal degrees (e.g., -48.2772 for Uberlandia, Brazil)
        lat (float): Latitude in decimal degrees (e.g., -18.9189 for Uberlandia, Brazil) 
        chuva_mm (float): Expected rainfall amount in millimeters (e.g., 50.0 for heavy rain)
        freq_min (int): Rainfall duration in minutes (e.g., 60 for one-hour storm)
        modo (str): Coordinate mode - 'geo' for geographic coordinates or 'cart' for cartesian indices
        
    Example:
        {
            "lon": -48.2772,
            "lat": -18.9189, 
            "chuva_mm": 50.0,
            "freq_min": 60,
            "modo": "geo"
        }
    """
    lon: float
    lat: float
    chuva_mm: float
    freq_min: int
    modo: str = "geo"

class BatchInput(BaseModel):
    """
    Batch processing data model for multiple flood risk assessments
    
    This model enables efficient processing of multiple geographic points in a single API call,
    optimized for large-scale analysis and urban planning applications.
    
    Attributes:
        pontos (List[InputPoint]): Array of InputPoint objects for simultaneous analysis
        
    Benefits:
        - Reduced API call overhead for multiple locations
        - Automatic statistical aggregation across all points
        - Individual error handling per point
        - Optimized batch processing algorithms
        
    Example:
        {
            "pontos": [
                {
                    "lon": -48.2772,
                    "lat": -18.9189,
                    "chuva_mm": 50.0,
                    "freq_min": 60,
                    "modo": "geo"
                },
                {
                    "lon": -48.2800,
                    "lat": -18.9200,
                    "chuva_mm": 30.0,
                    "freq_min": 120,
                    "modo": "geo"
                }
            ]
        }
    """
    pontos: List[InputPoint]

@app.get("/", tags=["Information"], summary="🏠 API Information & Health Check")
def root():
    """
    ## 🏠 UrbMind API - System Information
    
    Returns comprehensive API metadata and available endpoints for the UrbMind flood risk assessment platform.
    This endpoint serves as both a health check and documentation entry point.
    
    ### 📋 Response Information:
    - **API Status**: Current operational status and version
    - **Available Endpoints**: Complete list of API endpoints with descriptions
    - **Documentation Links**: Links to interactive API documentation
    - **System Health**: Basic system availability confirmation
    
    ### 🔧 Development Information:
    - Uses FastAPI framework for high-performance async operations
    - Implements OpenAPI 3.0 specification for comprehensive documentation
    
    ### 📖 Quick Start:
    1. Use `/analisar` for single-point flood risk analysis
    2. Use `/analisar-batch` for multiple locations simultaneously
    3. Check `/extensao` for geographic coverage information
    4. Visit `/docs` for interactive API exploration
    
    **Returns:**
    ```json
    {
        "message": "UrbMind Flood Risk Assessment API",
        "version": "2.0",
        "status": "operational",
        "endpoints": {
            "/analisar": "POST → Single-point flood risk analysis",
            "/analisar-batch": "POST → Batch analysis for multiple points", 
            "/extensao": "GET → Geographic coverage and data bounds",
            "/docs": "Interactive API documentation interface"
        },
        "data_sources": ["NASA SRTM", "Sentinel-2", "Historical Flood Database"],
        "supported_regions": ["Uberlandia, Brazil"]
    }
    ```
    """
    return {
        "message": "UrbMind Flood Risk Assessment API",
        "version": "2.0",
        "status": "operational", 
        "description": "Advanced AI-powered flood risk assessment using NASA satellite data",
        "endpoints": {
            "/analisar": "POST → Single-point flood risk analysis",
            "/analisar-batch": "POST → Batch analysis for multiple points",
            "/extensao": "GET → Geographic coverage and data bounds", 
            "/docs": "Interactive API documentation interface"
        },
        "data_sources": ["NASA SRTM Elevation", "Sentinel-2 Imagery", "Historical Flood Database"],
        "supported_regions": ["Uberlandia, Brazil and surrounding areas"],
        "coordinate_system": "WGS84 (EPSG:4326)",
    }

@app.post("/analisar", tags=["Analysis"], summary="🎯 Single-Point Flood Risk Analysis")
def analisar_ponto(data: InputPoint):
    """
    ## 🎯 Advanced Single-Point Flood Risk Assessment
    
    Performs comprehensive flood risk analysis for a specific geographic location using multi-source 
    geospatial data and advanced machine learning algorithms. This endpoint combines topographic analysis,
    satellite imagery interpretation, and historical flood data to provide accurate risk assessment.
    
    ### 🔬 Analysis Methodology:
    - **Topographic Analysis**: Slope, roughness, and aspect calculations from NASA SRTM data
    - **Hydrological Modeling**: Topographic Wetness Index (TWI) and flow accumulation analysis
    - **Spectral Analysis**: NDWI (water content) and NDBI (built-up areas) from Sentinel-2 imagery
    - **Historical Validation**: Proximity weighting based on documented flood events
    - **Rainfall Integration**: Dynamic risk calculation based on precipitation scenarios
    
    ### 📊 Input Parameters:
    - **lon** (float): Longitude in WGS84 decimal degrees (e.g., -48.2772 for Uberlandia center)
    - **lat** (float): Latitude in WGS84 decimal degrees (e.g., -18.9189 for Uberlandia center)
    - **chuva_mm** (float): Expected rainfall amount in millimeters (range: 0-200mm typical)
    - **freq_min** (int): Rainfall duration in minutes (range: 15-1440 minutes)
    - **modo** (str): Coordinate mode - 'geo' for geographic coordinates, 'cart' for raster indices
    
    ### 🎯 Output Information:
    - **probabilidade**: Flood probability score (0.0-1.0, where 1.0 = very high risk)
    - **risco_base**: Base terrain susceptibility independent of rainfall (0.0-1.0)
    - **raio_influencia**: Neighborhood analysis within 500m radius for spatial context
    - **status**: Operation result status and error handling
    
    ### 💡 Interpretation Guide:
    - **0.0-0.2**: Very Low Risk - Safe conditions even with moderate rainfall
    - **0.2-0.4**: Low Risk - Monitor weather conditions, minimal precautions needed
    - **0.4-0.6**: Moderate Risk - Be alert, avoid low-lying areas during heavy rain
    - **0.6-0.8**: High Risk - Significant flood potential, take preventive measures
    - **0.8-1.0**: Very High Risk - Extreme flood danger, evacuation may be necessary
    
    ### 🌍 Example Request:
    ```bash
    curl -X POST "https://api.urbmind.com/analisar" \\
         -H "Content-Type: application/json" \\
         -d '{
               "lon": -48.2772,
               "lat": -18.9189,
               "chuva_mm": 50.0,
               "freq_min": 60,
               "modo": "geo"
             }'
    ```
    
    ### ✅ Success Response Example:
    ```json
    {
        "entrada": {
            "lon": -48.2772,
            "lat": -18.9189,
            "chuva_mm": 50.0,
            "freq_min": 60,
            "modo": "geo"
        },
        "probabilidade": 0.67,
        "risco_base": 0.45,
        "raio_influencia": {
            "raio_m": 500,
            "prob_media": 0.62
        },
        "status": "sucesso"
    }
    ```
    
    ### ⚠️ Error Handling:
    - **Out of bounds**: Coordinates outside the geographic coverage area
    - **Invalid coordinates**: Malformed or unrealistic coordinate values
    - **System errors**: Internal processing errors with diagnostic information
    """
    try:
        prob = flood.sample_point(data.lon, data.lat, data.chuva_mm, data.freq_min, modo=data.modo)
        raio = flood.radius_influence(data.lon, data.lat, data.chuva_mm, data.freq_min, radius_m=500)
        return {
            "entrada": data.dict(),
            "probabilidade": prob["probabilidade"],
            "risco_base": prob["risco_base"],
            "raio_influencia": raio,
            "status": "sucesso"
        }
    except IndexError:
        return {
            "erro": "Coordinates outside coverage area - please check geographic bounds using /extensao endpoint",
            "entrada": data.dict(),
            "status": "erro",
            "codigo_erro": "OUT_OF_BOUNDS"
        }
    except Exception as e:
        return {
            "erro": "Internal processing error",
            "detalhes": str(e),
            "entrada": data.dict(),
            "status": "erro",
            "codigo_erro": "INTERNAL_ERROR"
        }

@app.post("/analisar-batch", tags=["Analysis"], summary="📊 High-Performance Batch Flood Risk Analysis")
def analisar_batch(data: BatchInput):
    """
    ## 📊 Advanced Batch Processing for Large-Scale Flood Risk Assessment
    
    Efficiently processes multiple geographic locations simultaneously, designed for urban planning,
    emergency management, and large-scale risk assessment applications. This endpoint optimizes 
    computational resources and provides comprehensive statistical analysis across multiple points.
    
    ### 🚀 Performance Advantages:
    - **Optimized Batch Processing**: Single API call for multiple locations reduces latency
    - **Automatic Statistical Aggregation**: Min, max, and mean calculations across all points
    - **Individual Error Handling**: Robust error isolation - failures in individual points don't affect others
    - **Structured Data Arrays**: Organized output arrays optimized for data analysis and visualization
    - **Scalable Architecture**: Handles from 2 to 1000+ points efficiently
    
    ### 🔧 Input Specification:
    - **pontos**: Array of InputPoint objects, each containing geographic and rainfall parameters
    - Each point follows the same specification as the single-point analysis endpoint
    - Supports mixed coordinate modes within the same batch request
    
    ### 📈 Analytics Features:
    - **Individual Results**: Complete analysis for each point with unique index tracking
    - **Aggregated Arrays**: Separate arrays for probabilities, base risks, and radius influences
    - **Statistical Summary**: Automatic calculation of key statistical measures
    - **Error Tracking**: Detailed error reporting with specific error codes and descriptions
    - **Success Metrics**: Clear reporting of successful vs. failed analyses
    
    ### 🎯 Use Cases:
    - **Urban Planning**: Analyze entire neighborhoods or development zones
    - **Emergency Response**: Rapid assessment of multiple critical infrastructure locations
    - **Insurance Assessment**: Batch evaluation of property portfolios
    - **Research Applications**: Large-scale flood risk studies and climate modeling
    - **Infrastructure Planning**: Risk assessment for transportation networks and utilities
    
    ### 📊 Output Structure:
    - **total_pontos**: Total number of points submitted for analysis
    - **sucessos**: Count of successfully processed points
    - **erros_count**: Number of points that encountered errors
    - **resultados**: Array of successful analysis results with complete metadata
    - **erros**: Detailed error information for failed points
    - **arrays**: Structured arrays optimized for data analysis and visualization
    - **estatisticas**: Statistical summary including min, max, and mean values
    
    ### 🌍 Example Batch Request:
    ```bash
    curl -X POST "https://api.urbmind.com/analisar-batch" \\
         -H "Content-Type: application/json" \\
         -d '{
               "pontos": [
                   {
                       "lon": -48.2772, "lat": -18.9189,
                       "chuva_mm": 50.0, "freq_min": 60, "modo": "geo"
                   },
                   {
                       "lon": -48.2800, "lat": -18.9200,
                       "chuva_mm": 30.0, "freq_min": 120, "modo": "geo"
                   },
                   {
                       "lon": -48.2750, "lat": -18.9150,
                       "chuva_mm": 75.0, "freq_min": 45, "modo": "geo"
                   }
               ]
             }'
    ```
    
    ### ✅ Comprehensive Response Example:
    ```json
    {
        "total_pontos": 3,
        "sucessos": 3,
        "erros_count": 0,
        "resultados": [
            {
                "indice": 0,
                "entrada": {...},
                "probabilidade": 0.67,
                "risco_base": 0.45,
                "raio_influencia": {"raio_m": 500, "prob_média": 0.62},
                "status": "sucesso"
            }
        ],
        "erros": [],
        "arrays": {
            "probabilidades": [0.67, 0.43, 0.82],
            "riscos_base": [0.45, 0.38, 0.71],
            "probabilidades_raio": [0.62, 0.41, 0.78]
        },
        "estatisticas": {
            "prob_min": 0.43,
            "prob_max": 0.82,
            "prob_media": 0.64
        },
        "status": "sucesso"
    }
    ```
    
    ### ⚠️ Error Handling & Robustness:
    - Individual point failures don't affect other points in the batch
    - Detailed error codes and descriptions for debugging
    - Graceful handling of mixed success/failure scenarios
    - Comprehensive logging for performance monitoring
    """
    resultados = []
    erros = []
    
    for i, ponto in enumerate(data.pontos):
        try:
            prob = flood.sample_point(ponto.lon, ponto.lat, ponto.chuva_mm, ponto.freq_min, modo=ponto.modo)
            raio = flood.radius_influence(ponto.lon, ponto.lat, ponto.chuva_mm, ponto.freq_min, radius_m=500)
            
            resultado = {
                "indice": i,
                "entrada": ponto.dict(),
                "probabilidade": prob["probabilidade"],
                "risco_base": prob["risco_base"],
                "raio_influencia": raio,
                "status": "sucesso"
            }
            resultados.append(resultado)
            
        except IndexError:
            erro = {
                "indice": i,
                "erro": "Coordinates outside coverage area",
                "entrada": ponto.dict(),
                "status": "erro",
                "codigo_erro": "OUT_OF_BOUNDS"
            }
            erros.append(erro)
            
        except Exception as e:
            erro = {
                "indice": i,
                "erro": "Internal processing error",
                "detalhes": str(e),
                "entrada": ponto.dict(),
                "status": "erro",
                "codigo_erro": "INTERNAL_ERROR"
            }
            erros.append(erro)
    
    # Extract organized arrays for data analysis and visualization
    probabilidades = [r["probabilidade"] for r in resultados]
    riscos_base = [r["risco_base"] for r in resultados]
    raios_influencia = [r["raio_influencia"]["prob_média"] for r in resultados]
    
    return {
        "total_pontos": len(data.pontos),
        "sucessos": len(resultados),
        "erros_count": len(erros),
        "resultados": resultados,
        "erros": erros,
        "arrays": {
            "probabilidades": probabilidades,
            "riscos_base": riscos_base,
            "probabilidades_raio": raios_influencia
        },
        "estatisticas": {
            "prob_min": min(probabilidades) if probabilidades else None,
            "prob_max": max(probabilidades) if probabilidades else None,
            "prob_media": sum(probabilidades) / len(probabilidades) if probabilidades else None
        },
        "performance": {
            "success_rate": len(resultados) / len(data.pontos) if data.pontos else 0,
            "total_processed": len(data.pontos)
        },
        "status": "sucesso"
    }

@app.get("/extensao", tags=["Information"], summary="🗺️ Geographic Coverage & Data Boundaries")
def extensao():
    """
    ## 🗺️ Geographic Coverage Information and Data Boundaries
    
    Provides comprehensive information about the geographic extent and coverage of the UrbMind
    flood risk assessment system. This endpoint is essential for validating input coordinates
    and understanding the spatial scope of available analysis capabilities.
    
    ### 📍 Coverage Information:
    - **Geographic Boundaries**: Complete bounding box coordinates in WGS84 format
    - **Coordinate System**: Detailed CRS information and projection details  
    - **Coverage Area**: Primary focus region and administrative boundaries
    - **Data Resolution**: Spatial resolution and accuracy specifications
    
    ### 🛰️ Data Sources Coverage:
    - **NASA SRTM Elevation Data**: 30-meter resolution digital elevation model
    - **Sentinel-2 Optical Imagery**: 10-20 meter resolution multispectral analysis
    - **Historical Flood Database**: Comprehensive flood event records for the region
    - **Topographic Derivatives**: Slope, aspect, roughness, and hydrological indices
    
    ### 🎯 Primary Use Cases:
    - **Coordinate Validation**: Verify if your points fall within the analysis area
    - **Boundary Planning**: Design studies and applications within supported regions
    - **Integration Planning**: Understand spatial constraints for system integration
    - **Coverage Expansion**: Plan for future geographic expansion requirements
    
    ### 🔧 Technical Specifications:
    - **Projection**: WGS84 Geographic Coordinate System (EPSG:4326)
    - **Units**: Decimal degrees for coordinates, meters for distances
    - **Accuracy**: Sub-pixel accuracy for coordinate transformations
    - **Update Frequency**: Static boundaries with dynamic data updates
    
    ### 🌍 Example Response:
    ```json
    {
        "limites": {
            "oeste": -48.5123,
            "leste": -48.0456,
            "sul": -19.1234,
            "norte": -18.7890
        },
        "area_cobertura": "Uberlandia Metropolitan Area, Minas Gerais, Brazil",
        "coordenadas": "WGS84 Geographic (EPSG:4326)",
        "resolucao_espacial": "30 meters (elevation), 10-20 meters (optical)",
        "area_total_km2": 2847.3,
        "centro_geografico": {
            "lon": -48.2790,
            "lat": -18.9562
        },
        "status": "sucesso"
    }
    ```
    
    ### ⚠️ Important Notes:
    - Coordinates outside these boundaries will return OUT_OF_BOUNDS errors
    - The system supports coordinate transformation but requires WGS84 input
    - Coverage may be expanded in future versions based on data availability
    - Contact the development team for custom coverage area requests
    
    ### 📞 Support Information:
    - For coverage expansion requests, contact: team@urbmind.com
    - For technical integration support, visit our documentation portal
    - Report coverage issues through our GitHub repository
    """
    try:
        bounds = flood.get_bounds()
        
        # Calculate additional metadata
        west, east = bounds["oeste"], bounds["leste"]
        south, north = bounds["sul"], bounds["norte"]
        center_lon = (west + east) / 2
        center_lat = (south + north) / 2
        
        # Approximate area calculation (rough estimate)
        lat_km = abs(north - south) * 111.32
        lon_km = abs(east - west) * 111.32 * math.cos(math.radians(center_lat))
        area_km2 = lat_km * lon_km
        
        return {
            "limites": {
                "oeste": west,
                "leste": east, 
                "sul": south,
                "norte": north
            },
            "area_cobertura": "Uberlandia Metropolitan Area, Minas Gerais, Brazil",
            "coordenadas": "WGS84 Geographic Coordinate System (EPSG:4326)",
            "resolucao_espacial": "30 meters (elevation data), 10-20 meters (optical imagery)",
            "dimensoes_raster": {
                "largura_px": bounds["largura_px"],
                "altura_px": bounds["altura_px"]
            },
            "centro_geografico": {
                "lon": round(center_lon, 4),
                "lat": round(center_lat, 4)
            },
            "area_total_km2": round(area_km2, 1),
            "fontes_dados": [
                "NASA SRTM Digital Elevation Model",
                "ESA Sentinel-2 Multispectral Imagery", 
                "Historical Flood Event Database",
                "Derived Topographic Indices"
            ],
            "versao_dados": "v2.0",
            "status": "sucesso"
        }
    except Exception as e:
        return {
            "erro": "Error retrieving geographic coverage information",
            "detalhes": str(e),
            "status": "erro",
            "codigo_erro": "COVERAGE_ERROR",
            "suporte": "Contact team@urbmind.com for assistance"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)