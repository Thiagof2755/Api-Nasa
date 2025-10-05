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

app = FastAPI(
    title="🌊 UrbMind - Flood Risk API",
    description="""
    ## API de Análise de Risco de Enchentes
    
    Esta API utiliza dados geoespaciais e machine learning para analisar o risco de enchentes em pontos específicos.
    
    ### Funcionalidades principais:
    - **Análise pontual**: Avalia risco de enchente para coordenadas específicas
    - **Análise em lote**: Processa múltiplos pontos simultaneamente
    - **Raio de influência**: Calcula impacto em área ao redor do ponto
    - **Dados geográficos**: Informações sobre extensão e cobertura
    
    ### Dados utilizados:
    - Modelo de elevação (SRTM)
    - Dados de declividade e rugosidade do terreno
    - Imagens de satélite Sentinel-2
    - Histórico de ocorrências de alagamentos
    """,
    version="2.0",
    contact={
        "name": "UrbMind Team",
        "email": "contato@urbmind.com",
    },
    license_info={
        "name": "MIT",
    },
)

class InputPoint(BaseModel):
    """
    Modelo para entrada de dados de um ponto específico
    
    - lon: Longitude em graus decimais (ex: -48.2772)
    - lat: Latitude em graus decimais (ex: -18.9189)
    - chuva_mm: Precipitação em milímetros (ex: 50.0)
    - freq_min: Frequência em minutos (ex: 60)
    - modo: Modo de coordenadas 'geo' ou 'cart' (padrão: 'geo')
    """
    lon: float
    lat: float
    chuva_mm: float
    freq_min: int
    modo: str = "geo"

class BatchInput(BaseModel):
    """
    Modelo para análise em lote de múltiplos pontos
    
    - pontos: Lista de pontos para análise
    """
    pontos: List[InputPoint]

@app.get("/", tags=["Informações"], summary="Informações da API")
def root():
    """
    Endpoint principal com informações sobre a API e seus endpoints disponíveis.
    
    Retorna:
    - Informações gerais da API
    - Lista de endpoints disponíveis
    - Links para documentação
    """
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

@app.post("/analisar", tags=["Análise"], summary="Análise de risco pontual")
def analisar_ponto(data: InputPoint):
    """
    Analisa o risco de enchente para um ponto específico.
    
    **Parâmetros:**
    - **lon**: Longitude em graus decimais (ex: -48.2772 para Uberlândia)
    - **lat**: Latitude em graus decimais (ex: -18.9189 para Uberlândia)
    - **chuva_mm**: Precipitação esperada em milímetros
    - **freq_min**: Frequência da precipitação em minutos
    - **modo**: Tipo de coordenadas ('geo' para geográficas, 'cart' para cartesianas)
    
    **Retorna:**
    - Probabilidade de enchente (0-1)
    - Risco base da região
    - Análise do raio de influência
    - Status da operação
    
    **Exemplo de uso:**
    ```json
    {
        "lon": -48.2772,
        "lat": -18.9189,
        "chuva_mm": 50.0,
        "freq_min": 60,
        "modo": "geo"
    }
    ```
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

@app.post("/analisar-batch", tags=["Análise"], summary="Análise de risco em lote")
def analisar_batch(data: BatchInput):
    """
    Analisa múltiplos pontos em lote e retorna arrays de resultados.
    
    **Vantagens da análise em lote:**
    - Processamento mais eficiente para múltiplos pontos
    - Estatísticas agregadas automáticas
    - Tratamento de erros individualizado
    - Arrays organizados para análise posterior
    
    **Parâmetros:**
    - **pontos**: Lista de objetos InputPoint para análise
    
    **Retorna:**
    - Resultados individuais para cada ponto
    - Arrays organizados de probabilidades e riscos
    - Estatísticas agregadas (min, max, média)
    - Contadores de sucessos e erros
    
    **Exemplo de uso:**
    ```json
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
    ```
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

@app.get("/extensao", tags=["Informações"], summary="Extensão geográfica dos dados")
def extensao():
    """
    Retorna informações sobre a extensão geográfica dos dados disponíveis.
    
    **Informações fornecidas:**
    - Limites geográficos (bounding box) dos dados
    - Sistema de coordenadas utilizado
    - Área de cobertura principal
    - Status da operação
    
    **Útil para:**
    - Verificar se suas coordenadas estão na área de cobertura
    - Entender os limites geográficos da análise
    - Validar dados de entrada antes de enviar
    
    **Retorna:**
    ```json
    {
        "limites": {
            "min_lon": -48.5,
            "max_lon": -48.0,
            "min_lat": -19.2,
            "max_lat": -18.7
        },
        "area_cobertura": "Uberlândia e região",
        "coordenadas": "EPSG:4326 (WGS84)",
        "status": "sucesso"
    }
    ```
    """
    try:
        bounds = flood.get_bounds()
        return {
            "limites": bounds,
            "area_cobertura": "Uberlândia e região",
            "coordenadas": "EPSG:4326 (WGS84)",
            "status": "sucesso"
        }
    except Exception as e:
        return {
            "erro": "Erro ao obter extensão",
            "detalhes": str(e),
            "status": "erro"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)