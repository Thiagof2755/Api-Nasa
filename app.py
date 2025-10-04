from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from model import FloodModel

# inicializa o modelo
flood = FloodModel(
    slope_tif="Dados/ImagenRelevos/viz.SRTMGL1_slope.tif",
    rough_tif="Dados/ImagenRelevos/viz.SRTMGL1_roughness.tif",
    aspect_tif="Dados/ImagenRelevos/viz.SRTMGL1_aspect.tif",
    sentinel_tif="Dados/Atmospheric_penetration/2025-09-30-00_00_2025-09-30-23_59_Sentinel-2_L2A_Atmospheric_Penetration.tiff"
)

app = FastAPI(title="Flood Risk API", version="2.0")

class InputPoint(BaseModel):
    lon: float
    lat: float
    chuva_mm: float
    freq_min: int
    modo: str = "geo"  # "geo" ou "cart"

class BatchInput(BaseModel):
    pontos: List[InputPoint]

@app.get("/")
def root():
    return {
        "mensagem": "API de Análise de Risco de Enchentes",
        "versao": "2.0",
        "endpoints": {
            "/analisar": "POST → analisa risco no ponto",
            "/analisar-batch": "POST → analisa múltiplos pontos em lote",
            "/extensao": "GET → mostra limites e tamanho do raster",
            "/docs": "interface interativa da API"
        }
    }

@app.post("/analisar")
def analisar_ponto(data: InputPoint):
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
            "erro": "Coordenadas fora da área de cobertura",
            "entrada": data.dict(),
            "status": "erro"
        }
    except Exception as e:
        return {
            "erro": "Erro interno",
            "detalhes": str(e),
            "entrada": data.dict(),
            "status": "erro"
        }

@app.post("/analisar-batch")
def analisar_batch(data: BatchInput):
    """Analisa múltiplos pontos em lote e retorna arrays de resultados"""
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
                "erro": "Coordenadas fora da área de cobertura",
                "entrada": ponto.dict(),
                "status": "erro"
            }
            erros.append(erro)
            
        except Exception as e:
            erro = {
                "indice": i,
                "erro": "Erro interno",
                "detalhes": str(e),
                "entrada": ponto.dict(),
                "status": "erro"
            }
            erros.append(erro)
    
    # Extrair arrays para facilitar análise
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
        "status": "sucesso"
    }