import devito
import numpy as np

from pylops.utils import dottest
from pylops.waveeqprocessing.twoway import AcousticWave2D, AcousticWave3D

devito.configuration["log-level"] = "ERROR"


par = {
    "ny": 10,
    "nx": 12,
    "nz": 20,
    "tn": 500,
    "dy": 3,
    "dx": 1,
    "dz": 2,
    "nr": 8,
    "ns": 2,
}

v0 = 2
y = np.arange(par["ny"]) * par["dy"]
x = np.arange(par["nx"]) * par["dx"]
z = np.arange(par["nz"]) * par["dz"]

sx = np.linspace(x.min(), x.max(), par["ns"])
sy = np.linspace(y.min(), y.max(), par["ns"])

rx = np.linspace(x.min(), x.max(), par["nr"])
ry = np.linspace(y.min(), y.max(), par["nr"])


def test_acwave2d():
    """Dot-test for AcousticWave2D operator"""
    Dop = AcousticWave2D(
        shape = (par["nx"], par["nz"]),
        origin = (0, 0),
        spacing = (par["dx"], par["dz"]),
        vp = np.ones((par["nx"], par["nz"])) * 2e3,
        src_x=sx,
        src_z = 5,
        rec_x = rx,
        rec_z = 5,
        t0 = 0.0,
        tn = par["tn"],
        src_type = "Ricker",
        space_order=4,
        nbl=30,
        f0=15,
        dtype="float32",
    )

    assert dottest(
        Dop, par["ns"] * par["nr"] * Dop.geometry.nt, par["nz"] * par["nx"], atol=1e-1
    )


def test_acwave3d():
    """Dot-test for AcousticWave2D operator"""
    Dop = AcousticWave3D(
        shape = (par["nx"], par["ny"], par["nz"]),
        origin = (0, 0, 0),
        spacing = (par["dx"], par["dy"], par["dz"]),
        vp = np.ones((par["nx"], par["ny"], par["nz"])) * 2e3,
        src_x=sx,
        src_y =sy,
        src_z = 5,
        rec_x = rx,
        rec_y = ry,
        rec_z = 5,
        t0 = 0.0,
        tn = par["tn"],
        src_type = "Ricker",
        space_order=4,
        nbl=30,
        f0=15,
        dtype="float32",
    )

    assert dottest(
        Dop, par["ns"] * par["nr"] * Dop.geometry.nt, par["nz"] * par["nx"] * par["ny"], atol=1e-1
    )