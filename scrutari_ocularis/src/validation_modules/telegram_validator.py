from enum import Enum, auto

class ResultadoValidacion(Enum):
    VALIDO = auto()
    ERROR_TOTAL_VOTANTES = auto()
    ERROR_SUMA_VOTOS = auto()

def validar_telegrama(votos_partidos, votos_en_blanco, votos_nulos, votos_recurridos, votos_impugnados, total_votantes):
    total_votos_partidos = sum(votos_partidos.values())
    total_votos = total_votos_partidos + votos_en_blanco + votos_nulos + votos_recurridos + votos_impugnados
    
    if total_votos != total_votantes:
        return ResultadoValidacion.ERROR_TOTAL_VOTANTES
    
    if total_votos_partidos + votos_en_blanco + votos_nulos != total_votantes:
        return ResultadoValidacion.ERROR_SUMA_VOTOS
    
    return ResultadoValidacion.VALIDO

