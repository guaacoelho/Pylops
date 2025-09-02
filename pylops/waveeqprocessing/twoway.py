__all__ = [
    "AcousticWave2D",
    "AcousticWave3D",
    "ElasticWave2D",
    "ElasticWave3D",
    "ViscoAcousticWave2D",
    "ViscoAcousticWave3D",
]

from .waveequations.acousticwave import _AcousticWave
from .waveequations.elasticwave import _ElasticWave
from .waveequations.viscoacousticwave import (
    _ViscoAcousticWave,
    _ViscoMultiparameterWave,
)

AcousticWave2D = _AcousticWave
AcousticWave3D = _AcousticWave
ElasticWave2D = _ElasticWave
ElasticWave3D = _ElasticWave
ViscoAcousticWave2D = _ViscoAcousticWave
ViscoAcousticWave3D = _ViscoAcousticWave
MultiparameterViscoAcoustic2D = _ViscoMultiparameterWave
MultiparameterViscoAcoustic3D = _ViscoMultiparameterWave
