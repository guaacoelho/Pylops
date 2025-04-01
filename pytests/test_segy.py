import numpy as np
import pytest

from pylops.waveeqprocessing.twoway import AcousticWave2D
from pylops.waveeqprocessing.segy import ReadSEGY2D


def get_parameters():
    nx = 170
    nz = 80

    spacing = (25., 25.)  # Grid spacing in m. The domain size is now 1km by 1km
    origin = (0., 0.)
    shape = (nx, nz)
    space_order = 4
    nbl = 40

    dtype = np.float32

    src_type = "Ricker"

    t0 = 0.
    f0 = 2  # 0.002 KHz

    vp_top = 1.5
    vp_bottom = 3.5

    nlayers = 3

    # define a velocity profile in km/s
    v = np.empty(shape, dtype=dtype)
    v[:] = vp_top  # Top velocity (background)
    vp_i = np.linspace(vp_top, vp_bottom, nlayers)
    for i in range(1, nlayers):
        v[..., i * int(shape[-1] / nlayers):] = vp_i[i]  # Bottom velocity

    return shape, spacing, origin, space_order, nbl, dtype, src_type, t0, f0, v


def create_operator_from_segy(segyReader, shot_id, shape, origin, spacing, v, nbl, space_order, t0, src_type, f0, dtype, tn, dt):
    """Cria uma instância de AcousticWave2D com base nos dados do SEGY."""
    sx, sz = segyReader.getSourceCoords(shot_id)
    rx, rz = segyReader.getReceiverCoords(shot_id)

    return AcousticWave2D(
        shape=shape, origin=origin, spacing=spacing, vp=v, nbl=nbl,
        space_order=space_order, src_x=sx, src_z=sz, rec_x=rx, rec_z=rz,
        t0=t0, tn=tn, src_type=src_type, f0=f0, dtype=dtype, dt=dt
    )


@pytest.mark.parametrize("chunk_size", [3, 6 , 10, 20, 39])
def test_getData_concatenation(chunk_size):
    path_segy = "/home/gustavo.coelho/segy-files/ModelShots/Anisotropic_FD_Model_Shots_part1.sgy"
    first_index = 1  # index do primeiro tiro do arquivo segy. A depender do arquivo pode variar começando em 0 ou em 1.
    segyReader = ReadSEGY2D(path_segy)

    expected_concatenation = [segyReader.getData(index=i + first_index, chunk_size=1) for i in range(chunk_size)]
    expected_concatenation = np.concatenate(expected_concatenation)

    # Retrieve concatenated data
    concatenated_data = segyReader.getData(index=0 + first_index, chunk_size=chunk_size)

    # Check if concatenation is as expected
    assert np.array_equal(concatenated_data, expected_concatenation), "Concatenation of data is not as expected"


@pytest.mark.parametrize("chunk_size, nshots", [(3, 3), (3, 6), (5, 19), (10, None)])
def test_getOperator(chunk_size, nshots):
    """
    Test the creation of operators from SEGY data.

    Parameters:
    chunk_size (int): Number of shots to process in each chunk.
    nshots (int or None): Total number of shots to process. If None, process all available shots.
    """
    path_segy = "/home/gustavo.coelho/segy-files/ModelShots/Anisotropic_FD_Model_Shots_part1.sgy"
    first_index = 1  # index do primeiro tiro do arquivo segy. A depender do arquivo pode variar começando em 0 ou em 1.
    segyReader = ReadSEGY2D(path_segy)

    # Obtém parâmetros do modelo
    shape, spacing, origin, space_order, nbl, dtype, src_type, t0, f0, v = get_parameters()

    # Criação dos operadores a partir do SEGY
    operator_params = {
        "shape": shape, "origin": origin, "spacing": spacing, "vp0": v,
        "nbl": nbl, "space_order": space_order, "t0": t0, "src_type": src_type,
        "f0": f0, "dtype": dtype, "chunk": chunk_size, "nshots": nshots, "src0_idx": first_index
    }
    Aops = segyReader.getOperator(**operator_params)

    dt = segyReader.getDt()
    tn = segyReader.getTn()

    end_idx = nshots + first_index if nshots else segyReader.nsrc
    # Criação manual dos operadores de referência
    reference_operators = [create_operator_from_segy(segyReader, shot_id, shape, origin, spacing,
                                                     v, nbl, space_order, t0, src_type, f0, dtype, tn, dt)
                           for shot_id in range(0 + first_index, end_idx)]

    # Teste de consistência entre os operadores retornados e os operadores esperados
    for i, Aop in enumerate(Aops):
        for j in range(len(Aop.ops)):
            assert np.array_equal(Aop.ops[j].geometry.rec_positions, reference_operators[i * chunk_size + j].geometry.rec_positions), f"Falha no teste do operador {i}, shot {j}"
