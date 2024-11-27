__all__ = [
    "AcousticWave2D",
    "AcousticWave3D",
    "ElasticWave2D",
    "ElasticWave3D",
    "ViscoAcousticWave2D",
    "ViscoAcousticWave3D",
]

from copy import deepcopy
from typing import Tuple, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike

devito_message = deps.devito_import("the twoway module")

if devito_message is None:
    from devito import Function
    from devito.builtins import initialize_function

    from examples.seismic import AcquisitionGeometry, Model, Receiver
    from examples.seismic.acoustic import AcousticWaveSolver
    from examples.seismic.source import TimeAxis
    from examples.seismic.stiffness import IsoElasticWaveSolver, ISOSeismicModel
    from examples.seismic.utils import PointSource, sources
    from examples.seismic.viscoacoustic import ViscoacousticWaveSolver


class _CustomSource(PointSource):
    """Custom source

    This class creates a Devito symbolic object that encapsulates a set of
    sources with a user defined source signal wavelet ``wav``

    Parameters
    ----------
    name : :obj:`str`
        Name for the resulting symbol.
    grid : :obj:`devito.types.grid.Grid`
        The computational domain.
    time_range : :obj:`examples.seismic.source.TimeAxis`
        TimeAxis(start, step, num) object.
    wav : :obj:`numpy.ndarray`
        Wavelet of size

    """

    __rkwargs__ = PointSource.__rkwargs__ + ["wav"]

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        kwargs.setdefault("npoint", 1)

        return super().__args_setup__(*args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super().__init_finalize__(*args, **kwargs)

        self.wav = kwargs.get("wav")

        if not self.alias:
            for p in range(kwargs["npoint"]):
                self.data[:, p] = self.wavelet

    @property
    def wavelet(self):
        """Return user-provided wavelet"""
        return self.wav


class _AcousticWave(LinearOperator):
    """Devito Acoustic propagator.

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple` or  :obj:`numpy.ndarray`
        Model spacing ``(dx, dz)``
    vp : :obj:`numpy.ndarray`
        Velocity model in m/s
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_y : :obj:`numpy.ndarray`
        Source y-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_y : :obj:`numpy.ndarray`
        Receiver y-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time in ms
    tn : :obj:`int`
        Final time in ms
    src_type : :obj:`str`
        Source type
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
        but it will slow down computations
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        src_type: str = "Ricker",
        space_order: int = 6,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        op_name: str = "born",
        src_y: NDArray = None,
        rec_y: NDArray = None,
        dt: int = None,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_folder_path: str = None,
        dswap_compression: str = None,
        dswap_compression_value: float | int = None,
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        # create model
        self._create_model(shape, origin, spacing, vp, space_order, nbl, dt)
        self._create_geometry(
            src_x, src_y, src_z, rec_x, rec_y, rec_z, t0, tn, src_type, f0=f0
        )
        self.checkpointing = checkpointing
        self.karguments = {}
        self._dswap_opt = {
            "dswap": dswap,
            "dswap_disks": dswap_disks,
            "dswap_folder": dswap_folder,
            "dswap_folder_path": dswap_folder_path,
            "dswap_compression": dswap_compression,
            "dswap_compression_value": dswap_compression_value,
        }

        super().__init__(
            dtype=np.dtype(dtype),
            dims=vp.shape,
            dimsd=(len(src_x), len(rec_x), self.geometry.nt),
            explicit=False,
            name=name,
        )
        self._register_multiplications(op_name)

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        space_order: int = 6,
        nbl: int = 20,
        dt: int = None,
    ) -> None:
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in m/s
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries

        """
        self.space_order = space_order
        self.model = Model(
            space_order=space_order,
            vp=vp * 1e-3,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
            dt=dt,
        )

    def _create_geometry(
        self,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        src_type: str,
        f0: float = 20.0,
    ) -> None:
        """Create geometry and time axis

        Parameters
        ----------
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in m
        src_y : :obj:`numpy.ndarray`
            Source y-coordinates in m
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in m
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in m
        rec_y : :obj:`numpy.ndarray`
            Receiver y-coordinates in m
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in m
        t0 : :obj:`float`
            Initial time
        tn : :obj:`int`
            Final time in ms
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, self.model.dim))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, -1] = src_z
        if self.model.dim == 3:
            src_coordinates[:, 1] = src_y

        rec_coordinates = np.empty((nrec, self.model.dim))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, -1] = rec_z
        if self.model.dim == 3:
            rec_coordinates[:, 1] = rec_y

        self.geometry = AcquisitionGeometry(
            self.model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 * 1e-3,
        )

    def updatesrc(self, wav):
        """Update source wavelet

        This routines is used to allow users to pass a custom source
        wavelet to replace the source wavelet generated when the
        object is initialized

        Parameters
        ----------
        wav : :obj:`numpy.ndarray`
            Wavelet

        """
        wav_padded = np.pad(wav, (0, self.geometry.nt - len(wav)))

        self.wav = _CustomSource(
            name="src",
            grid=self.model.grid,
            wav=wav_padded,
            time_range=self.geometry.time_axis,
        )

    def _srcillumination_oneshot(self, isrc: int) -> Tuple[NDArray, NDArray]:
        """Source wavefield and illumination for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model

        Returns
        -------
        u0 : :obj:`np.ndarray`
            Source wavefield
        src_ill : :obj:`np.ndarray`
            Source illumination

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[isrc, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )
        solver = AcousticWaveSolver(
            self.model, geometry, space_order=self.space_order, **self._dswap_opt
        )

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]

        # source wavefield
        u0 = solver.forward(
            save=True, src=None if not hasattr(self, "wav") else self.wav
        )[1]

        # source illumination
        src_ill = self._crop_model((u0.data**2).sum(axis=0), self.model.nbl)
        return u0, src_ill

    def srcillumination_allshots(self, savewav: bool = False) -> None:
        """Source wavefield and illumination for all shots

        Parameters
        ----------
        savewav : :obj:`bool`, optional
            Save source wavefield (``True``) or not (``False``)

        """
        nsrc = self.geometry.src_positions.shape[0]
        if savewav:
            self.src_wavefield = []
        self.src_illumination = np.zeros(self.model.shape)

        for isrc in range(nsrc):
            src_wav, src_ill = self._srcillumination_oneshot(isrc)
            if savewav:
                self.src_wavefield.append(src_wav)
            self.src_illumination += src_ill

    def _born_oneshot(self, solver: AcousticWaveSolver, dm: NDArray) -> NDArray:
        """Born modelling for one shot

        Parameters
        ----------
        solver : :obj:`AcousticWaveSolver`
            Devito's solver object.
        dm : :obj:`np.ndarray`
            Model perturbation

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """

        # set perturbation
        dmext = np.zeros(self.model.grid.shape, dtype=np.float32)
        if dmext.ndim == 2:
            dmext[
                self.model.nbl : -self.model.nbl,
                self.model.nbl : -self.model.nbl,
            ] = dm
        else:
            dmext[
                self.model.nbl : -self.model.nbl,
                self.model.nbl : -self.model.nbl,
                self.model.nbl : -self.model.nbl,
            ] = dm

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = solver.geometry.src_positions[:]

        d = solver.jacobian(dmext, src=None if not hasattr(self, "wav") else self.wav)[
            0
        ]
        d = d.resample(solver.geometry.dt).data[:][: solver.geometry.nt].T
        return d

    def _born_allshots(self, dm: NDArray) -> NDArray:
        """Born modelling for all shots

        Parameters
        -----------
        dm : :obj:`np.ndarray`
            Model perturbation

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        # solve
        solver = AcousticWaveSolver(
            self.model, geometry, space_order=self.space_order, **self._dswap_opt
        )

        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._born_oneshot(solver, dm)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _bornadj_oneshot(self, solver: AcousticWaveSolver, isrc, dobs):
        """Adjoint born modelling for one shot

        Parameters
        ----------
        isrc : :obj:`float`
            Index of source to model
        dobs : :obj:`np.ndarray`
            Observed data to inject

        Returns
        -------
        model : :obj:`np.ndarray`
            Model

        """
        # set disk_swap bool
        dswap = self._dswap_opt.get("dswap", False)

        # create boundary data
        recs = self.geometry.rec.copy()
        recs.data[:] = dobs.T[:]

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]

        # source wavefield
        if hasattr(self, "src_wavefield"):
            u0 = self.src_wavefield[isrc]
        else:
            u0 = solver.forward(
                save=True if not dswap else False,
                src=None if not hasattr(self, "wav") else self.wav,
            )[1]

        # adjoint modelling (reverse wavefield plus imaging condition)
        model = solver.jacobian_adjoint(
            rec=recs, u=u0, checkpointing=self.checkpointing
        )[0]
        return model

    def _bornadj_allshots(self, dobs: NDArray) -> NDArray:
        """Adjoint Born modelling for all shots

        Parameters
        ----------
        dobs : :obj:`np.ndarray`
            Observed data to inject

        Returns
        -------
        model : :obj:`np.ndarray`
            Model

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        solver = AcousticWaveSolver(
            self.model, geometry, space_order=self.space_order, **self._dswap_opt
        )

        nsrc = self.geometry.src_positions.shape[0]
        mtot = np.zeros(self.model.shape, dtype=np.float32)

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            m = self._bornadj_oneshot(solver, isrc, dobs[isrc])
            mtot += self._crop_model(m.data, self.model.nbl)
        return mtot

    def _fwd_oneshot(self, solver: AcousticWaveSolver, v: NDArray) -> NDArray:
        """Forward modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """
        # create function representing the physical parameter received as parameter
        function = Function(
            name="vp",
            grid=self.model.grid,
            space_order=self.model.space_order,
            parameter=True,
        )

        # Assignment of values to physical parameters functions based on the values in 'v'
        initialize_function(function, v, self.model.padsizes)

        # add vp to karguments to be used inside devito's solver
        self.karguments.update({"vp": function})

        d = solver.forward(**self.karguments)[0]
        d = d.resample(solver.geometry.dt).data[:][: solver.geometry.nt].T
        return d

    def _fwd_allshots(self, v: NDArray) -> NDArray:
        """Forward modelling for all shots

        Parameters
        -----------
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        # solve
        solver = AcousticWaveSolver(
            self.model,
            geometry,
            space_order=self.space_order,
        )

        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._fwd_oneshot(solver, v)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _register_multiplications(self, op_name: str) -> None:
        if op_name == "born":
            self._acoustic_matvec = self._born_allshots
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
        self._acoustic_rmatvec = self._bornadj_allshots

    def create_receiver(
        self, name, rx=None, ry=None, rz=None, t0=None, tn=None, dt=None
    ):
        if self.model.dim == 2 and ry is not None:
            raise Exception("Attempting to create 3D receiver for a 2D operator!")

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt

        rx = rx if rx is not None else self.geometry.rec_positions[:, 0]
        rz = rz if rz is not None else self.geometry.rec_positions[:, -1]
        if self.model.dim == 3:
            ry = ry if ry is not None else self.geometry.rec_positions[:, 1]

        nrec = len(rx)

        rec_coordinates = np.empty((nrec, self.model.dim))
        rec_coordinates[:, 0] = rx
        rec_coordinates[:, -1] = rz
        if self.model.dim == 3:
            rec_coordinates[:, 1] = ry

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)
        return Receiver(
            name=name,
            grid=self.geometry.grid,
            time_range=time_axis,
            npoint=nrec,
            coordinates=rec_coordinates,
        )

    def create_source(
        self,
        name,
        sx=None,
        sy=None,
        sz=None,
        t0=None,
        tn=None,
        dt=None,
        f0=None,
        src_type=None,
    ):

        if self.model.dim == 2 and sy is not None:
            raise Exception("Attempting to create 3D source for a 2D operator!")

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt
        f0 = f0 or self.geometry.f0

        src_type = src_type or self.geometry.src_type

        sx = sx or self.geometry.src_positions[:, 0]
        sz = sz or self.geometry.src_positions[:, -1]
        if self.model.dim == 3:
            sy = sy or self.geometry.src_positions[:, 1]

        nsrc = len(sx)

        src_coordinates = np.empty((nsrc, 3))
        src_coordinates[:, 0] = sx
        src_coordinates[:, -1] = sz
        if self.model.dim == 3:
            src_coordinates[:, 1] = sy

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)

        return sources[src_type](
            name=name,
            grid=self.geometry.grid,
            f0=f0,
            time_range=time_axis,
            npoint=nsrc,
            coordinates=src_coordinates,
            t0=t0,
        )

    def add_args(self, **kwargs):
        self.karguments = kwargs

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_matvec(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_rmatvec(x)
        return y


class AcousticWave2D(_AcousticWave):
    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        op_name: str = "born",
        dt: int = None,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_folder_path: str = None,
        dswap_compression: str = None,
        dswap_compression_value: float | int = None,
    ) -> None:

        if len(shape) != 2:
            raise Exception(
                "Attempting to create a 3D operator using a 2D intended class!"
            )

        super().__init__(
            shape=shape,
            origin=origin,
            spacing=spacing,
            vp=vp,
            src_x=src_x,
            src_z=src_z,
            rec_x=rec_x,
            rec_z=rec_z,
            t0=t0,
            tn=tn,
            src_type=src_type,
            space_order=space_order,
            nbl=nbl,
            f0=f0,
            checkpointing=checkpointing,
            dtype=dtype,
            name=name,
            op_name=op_name,
            dt=dt,
            dswap=dswap,
            dswap_disks=dswap_disks,
            dswap_folder=dswap_folder,
            dswap_folder_path=dswap_folder_path,
            dswap_compression=dswap_compression,
            dswap_compression_value=dswap_compression_value,
        )

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl]


class AcousticWave3D(_AcousticWave):
    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        op_name: str = "born",
        dt: int = None,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_folder_path: str = None,
        dswap_compression: str = None,
        dswap_compression_value: float | int = None,
    ) -> None:

        if len(shape) != 3:
            raise Exception(
                "Attempting to create a 3D operator with a 2D intended class!"
            )

        super().__init__(
            shape=shape,
            origin=origin,
            spacing=spacing,
            vp=vp,
            src_x=src_x,
            src_y=src_y,
            src_z=src_z,
            rec_x=rec_x,
            rec_y=rec_y,
            rec_z=rec_z,
            t0=t0,
            tn=tn,
            src_type=src_type,
            space_order=space_order,
            nbl=nbl,
            f0=f0,
            checkpointing=checkpointing,
            dtype=dtype,
            name=name,
            op_name=op_name,
            dt=dt,
            dswap=dswap,
            dswap_disks=dswap_disks,
            dswap_folder=dswap_folder,
            dswap_folder_path=dswap_folder_path,
            dswap_compression=dswap_compression,
            dswap_compression_value=dswap_compression_value,
        )

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl, nbl:-nbl]


class _ElasticWave(LinearOperator):
    """Devito Elastic propagator.

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple` or  :obj:`numpy.ndarray`
        Model spacing ``(dx, dz)``
    vp : :obj:`numpy.ndarray`
        Velocity model in m/s
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_y : :obj:`numpy.ndarray`
        Source y-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_y : :obj:`numpy.ndarray`
        Receiver y-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time
    tn : :obj:`int`
        Number of time samples
    src_type : :obj:`str`
        Source type
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
        but it will slow down computations
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    """

    _list_par = {
        "lam-mu": ["lam", "mu", "rho"],
        "vp-vs-rho": ["vp", "vs", "rho"],
        "Ip-Is-rho": ["Ip", "Is", "rho"],
    }

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        vs: NDArray,
        rho: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        par: str = "lam-mu",
        op_name: str = "fwd",
        dt: int = None,
        src_y: NDArray = None,
        rec_y: NDArray = None,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_folder_path: str = None,
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        # create model
        self._create_model(shape, origin, spacing, vp, vs, rho, space_order, nbl, dt)
        self._create_geometry(
            src_x, src_y, src_z, rec_x, rec_y, rec_z, t0, tn, src_type, f0=f0
        )
        self.checkpointing = checkpointing
        self.par = par
        self.karguments = {}
        dim = self.model.dim
        self._dswap_opt = {
            "dswap": dswap,
            "dswap_disks": dswap_disks,
            "dswap_folder": dswap_folder,
            "dswap_folder_path": dswap_folder_path,
        }

        n_input = 3
        num_outs = dim + 1
        dims = (n_input, self.model.vp.shape[0], self.model.vp.shape[1])
        # If dim is 3, add the last dimension
        if dim == 3:
            dims = (*dims, self.model.vp.shape[2])

        super().__init__(
            dtype=np.dtype(dtype),
            dims=dims,
            dimsd=(num_outs, len(src_x), len(rec_x), self.geometry.nt),
            explicit=False,
            name=name,
        )

        self._register_multiplications(op_name)

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        vs: NDArray,
        rho: NDArray,
        space_order: int = 6,
        nbl: int = 20,
        dt: int = None,
    ) -> None:
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in m/s
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries

        """
        self.space_order = space_order
        self.model = ISOSeismicModel(
            space_order=space_order,
            vp=vp / 1000,
            vs=vs / 1000,
            rho=rho,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
            dt=dt,
        )

    def _create_geometry(
        self,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str,
        f0: float = 20.0,
    ) -> None:
        """Create geometry and time axis

        Parameters
        ----------
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in m
        src_y : :obj:`numpy.ndarray`
            Source y-coordinates in m
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in m
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in m
        rec_y : :obj:`numpy.ndarray`
            Receiver y-coordinates in m
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in m
        t0 : :obj:`float`
            Initial time
        tn : :obj:`int`
            Number of time samples
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, self.model.dim))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, -1] = src_z
        if self.model.dim == 3:
            src_coordinates[:, 1] = src_y

        rec_coordinates = np.empty((nrec, self.model.dim))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, -1] = rec_z
        if self.model.dim == 3:
            rec_coordinates[:, 1] = rec_y

        self.geometry = AcquisitionGeometry(
            self.model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 / 1000,
        )

    def _fwd_oneshot(self, solver: IsoElasticWaveSolver, v: NDArray) -> NDArray:
        """Forward modelling for one shot

        Parameters
        ----------
        solver : :obj:`IsoElasticWaveSolver`
            Devito's solver object.
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """
        # If "par" was not provided as a parameter to forward execution, use the operator's default value
        self.karguments["par"] = self.karguments.get("par", self.par)

        # get arguments that will be used for this elastic forward execution
        args = self._list_par[self.karguments["par"]]

        # create functions representing the physical parameters received as parameters
        functions = [
            Function(
                name=name,
                grid=self.model.grid,
                space_order=self.model.space_order,
                parameter=True,
            )
            for name in args
        ]

        # Assignment of values to physical parameters functions based on the values in 'v'
        for function, value in zip(functions, v):
            initialize_function(function, value, 0)

        # Update 'karguments' to contain the values of the parameters defined in 'args'
        self.karguments.update(dict(zip(args, functions)))

        dim = self.model.dim
        rec_data = list(solver.forward(**self.karguments)[0 : dim + 1])

        for ii, d in enumerate(rec_data):
            rec_data[ii] = (
                d.resample(solver.geometry.dt).data[:][: solver.geometry.nt].T
            )
        return rec_data

    def _fwd_allshots(self, v: NDArray) -> NDArray:
        """Forward modelling for all shots

        Parameters
        -----------
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        # create solver
        solver = IsoElasticWaveSolver(
            self.model, geometry, space_order=self.space_order
        )

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._fwd_oneshot(solver, v)
            dtot.append(deepcopy(d))

        # Adjust dimensions
        rec_data = list(zip(*dtot))

        return np.array(rec_data)

    def _grad_oneshot(self, isrc, dobs, solver: IsoElasticWaveSolver):
        """Adjoint gradient modelling for one shot

        Parameters
        ----------
        isrc : :obj:`float`
            Index of source to model
        dobs : :obj:`np.ndarray`
            Observed data to inject
        solver : :obj:`IsoElasticWaveSolver`
            Devito's solver object

        Returns
        -------
        model : :obj:`np.ndarray`
            Model

        """
        dim = self.model.dim
        # create boundary data
        rec_vx = self.geometry.rec.copy()
        rec_vx.data[:] = dobs[1].T[:]
        rec_vz = self.geometry.rec.copy()
        rec_vz.data[:] = dobs[-1].T[:]
        if dim == 3:
            rec_vy = self.geometry.rec.copy()
            rec_vy.data[:] = dobs[2].T[:]

        if "rec_p" in self.karguments:
            # If it exists in the karguments, I update the rec_p data field in the karguments.
            self.karguments["rec_p"].data[:] = dobs[0].T[:]
        else:
            # If it does not exist in karguments, I copy the structure of rec, assign the data, and add it to the karguments
            rec_p = self.geometry.rec.copy()
            rec_p.data[:] = dobs[0].T[:]
            self.karguments["rec_p"] = rec_p

        # set disk_swap bool
        dswap = self._dswap_opt.get("dswap", False)

        # If "par" was not passed as a parameter to forward execution, use the operator's default value
        self.karguments["par"] = self.karguments.get("par", self.par)

        # source wavefield
        if hasattr(self, "src_wavefield"):
            u0 = self.src_wavefield[isrc]
        else:
            par = self.karguments.get("par")
            u0 = solver.forward(save=True if not dswap else False, par=par)[dim + 1]

        # adjoint modelling (reverse wavefield plus imaging condition)
        grad1, grad2, grad3 = solver.jacobian_adjoint(
            rec_vx,
            rec_vz,
            u0,
            rec_vy=None if dim == 2 else rec_vy,
            checkpointing=self.checkpointing,
            **self.karguments,
        )[0:3]

        return grad1, grad2, grad3

    def _grad_allshots(self, dobs: NDArray) -> NDArray:
        """Adjoint Gradient modelling for all shots

        Parameters
        ----------
        dobs : :obj:`np.ndarray`
            Observed data to inject.

            The shape of dobs is (n_input, nsrc, nrecs. nt):
            If it is 2-dimensional, the positions are as follows:
                dobs[0] = rec_tau
                dobs[1] = rec_vx
                dobs[2] = rec_vz

            And if 3-dimensional:
                dobs[0] = rec_tau
                dobs[1] = rec_vx
                dobs[2] = rec_vy
                dobs[3] = rec_vz

        Returns
        -------
        model : :obj:`np.ndarray`
            Model

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        nsrc = self.geometry.src_positions.shape[0]

        shape = self.model.grid.shape
        if self.model.dim == 2:
            mtot = np.zeros((3, shape[0], shape[1]), dtype=np.float32)
        elif self.model.dim == 3:
            mtot = np.zeros((3, shape[0], shape[1], shape[2]), dtype=np.float32)

        solver = IsoElasticWaveSolver(
            self.model, geometry, space_order=self.space_order, **self._dswap_opt
        )

        for isrc in range(nsrc):
            # For each dobs get data equivalent to isrc shot
            isrc_rec = [rec[isrc] for rec in dobs]

            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            grads = self._grad_oneshot(isrc, isrc_rec, solver)

            # post-process data
            for ii, g in enumerate(grads):
                mtot[ii] += g.data
        return mtot

    def _register_multiplications(self, op_name: str) -> None:
        self.op_name = op_name
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
            self._acoustic_rmatvec = self._grad_allshots
        else:
            raise Exception("The operator's name '%s' is not valid." % op_name)

    def create_receiver(
        self, name, rx=None, ry=None, rz=None, t0=None, tn=None, dt=None
    ):

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt

        rx = rx if rx is not None else self.geometry.rec_positions[:, 0]
        rz = rz if rz is not None else self.geometry.rec_positions[:, -1]

        if self.model.dim == 3:
            ry = ry if ry is not None else self.geometry.rec_positions[:, 1]

        nrec = len(rx)

        rec_coordinates = np.empty((nrec, self.model.dim))
        rec_coordinates[:, 0] = rx
        rec_coordinates[:, -1] = rz
        if self.model.dim == 3:
            rec_coordinates[:, 1] = ry

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)
        return Receiver(
            name=name,
            grid=self.geometry.grid,
            time_range=time_axis,
            npoint=nrec,
            coordinates=rec_coordinates,
        )

    def create_source(
        self,
        name,
        sx=None,
        sy=None,
        sz=None,
        t0=None,
        tn=None,
        dt=None,
        f0=None,
        src_type=None,
    ):

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt
        f0 = f0 or self.geometry.f0

        src_type = src_type or self.geometry.src_type

        sx = sx or self.geometry.src_positions[:, 0]
        sz = sz or self.geometry.src_positions[:, -1]
        if self.model.dim == 3:
            sy = sy or self.geometry.src_positions[:, 1]

        nsrc = len(sx)

        src_coordinates = np.empty((nsrc, self.model.dim))
        src_coordinates[:, 0] = sx
        src_coordinates[:, -1] = sz
        if self.model.dim == 2:
            src_coordinates[:, 1] = sy

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)

        return sources[src_type](
            name=name,
            grid=self.geometry.grid,
            f0=f0,
            time_range=time_axis,
            npoint=nsrc,
            coordinates=src_coordinates,
            t0=t0,
        )

    def add_args(self, **kwargs):
        # TODO: decide if this values will be manteined at the object or it will be resete after matvec's execution.
        self.karguments = deepcopy(kwargs)

    def __mul__(self, x: Union[float, LinearOperator]) -> LinearOperator:
        # data must be a np.array
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return super().dot(x)

    def forward(self, x: NDArray, **kwargs):
        # save current op_name to get back to it after the forward modelling
        save_op_name = self.op_name

        # Update operation's type forward
        self._register_multiplications("fwd")

        # Add arguments to self and execute _matvec
        self.add_args(**kwargs)
        y = self._matvec(x)

        # Reshape data to dimsd format
        y = y.reshape(getattr(self, "dimsd"))

        # Restore operation's type that was used before this forward modelling
        self._register_multiplications(save_op_name)
        return y

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_matvec(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_rmatvec(x)
        return y


class ElasticWave2D(_ElasticWave):
    """Devito Elastic propagator.

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple` or  :obj:`numpy.ndarray`
        Model spacing ``(dx, dz)``
    vp : :obj:`numpy.ndarray`
        Velocity model in m/s
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time
    tn : :obj:`int`
        Number of time samples
    src_type : :obj:`str`
        Source type
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
        but it will slow down computations
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        vs: NDArray,
        rho: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        par: str = "lam-mu",
        op_name: str = "fwd",
        dt: int = None,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_folder_path: str = None,
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        if len(shape) != 2:
            raise Exception(
                "Attempting to create a 3D operator using a 2D intended class!"
            )

        super().__init__(
            shape=shape,
            origin=origin,
            spacing=spacing,
            vp=vp,
            vs=vs,
            rho=rho,
            src_x=src_x,
            src_z=src_z,
            rec_x=rec_x,
            rec_z=rec_z,
            t0=t0,
            tn=tn,
            src_type=src_type,
            space_order=space_order,
            nbl=nbl,
            f0=f0,
            checkpointing=checkpointing,
            dtype=dtype,
            name=name,
            par=par,
            op_name=op_name,
            dt=dt,
            dswap=dswap,
            dswap_disks=dswap_disks,
            dswap_folder=dswap_folder,
            dswap_folder_path=dswap_folder_path,
        )

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl]


class ElasticWave3D(_ElasticWave):
    """Devito Elastic propagator.

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple` or  :obj:`numpy.ndarray`
        Model spacing ``(dx, dz)``
    vp : :obj:`numpy.ndarray`
        Velocity model in m/s
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_y : :obj:`numpy.ndarray`
        Source y-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_y : :obj:`numpy.ndarray`
        Receiver y-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time
    tn : :obj:`int`
        Number of time samples
    src_type : :obj:`str`
        Source type
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
        but it will slow down computations
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        vs: NDArray,
        rho: NDArray,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        par: str = "lam-mu",
        op_name: str = "fwd",
        dt: int = None,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_folder_path: str = None,
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        if len(shape) != 3:
            raise Exception(
                "Attempting to create a 2D operator with a 3D intended class!"
            )

        super().__init__(
            shape=shape,
            origin=origin,
            spacing=spacing,
            vp=vp,
            vs=vs,
            rho=rho,
            src_x=src_x,
            src_y=src_y,
            src_z=src_z,
            rec_x=rec_x,
            rec_y=rec_y,
            rec_z=rec_z,
            t0=t0,
            tn=tn,
            src_type=src_type,
            space_order=space_order,
            nbl=nbl,
            f0=f0,
            checkpointing=checkpointing,
            dtype=dtype,
            name=name,
            par=par,
            op_name=op_name,
            dt=dt,
            dswap=dswap,
            dswap_disks=dswap_disks,
            dswap_folder=dswap_folder,
            dswap_folder_path=dswap_folder_path,
        )

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl, nbl:-nbl]


class _ViscoAcousticWave(LinearOperator):
    """Devito ViscoAcoustic propagator.

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple` or  :obj:`numpy.ndarray`
        Model spacing ``(dx, dz)``
    vp : :obj:`numpy.ndarray`
        Velocity model in m/s
    qp : :obj:`numpy.ndarray
        P-wave attenuation
    b : :obj:`numpy.ndarray
        Buoyancy
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_y : :obj:`numpy.ndarray`
        Source y-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_y : :obj:`numpy.ndarray`
        Receiver y-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time
    tn : :obj:`int`
        Number of time samples
    src_type : :obj:`str`
        Source type
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    kernel :
        selects a visco-acoustic equation from the options below:
                'sls' (Standard Linear Solid) :
                1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
                viscoacoustic equation
                2nd order - Bai et al. (2014) viscoacoustic equation
                'kv' - Ren et al. (2014) viscoacoustic equation
                'maxwell' - Deng and McMechan (2007) viscoacoustic equation
                Defaults to 'sls' 2nd order.
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
        but it will slow down computations
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        qp: NDArray,
        b: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        kernel: str = "sls",
        time_order: int = 2,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        op_name: str = "fwd",
        src_y: NDArray = None,
        rec_y: NDArray = None,
        dt: int = None,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_folder_path: str = None,
        dswap_compression: str = None,
        dswap_compression_value: float | int = None,
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        # create model
        self._create_model(shape, origin, spacing, vp, qp, b, space_order, nbl, dt)
        self._create_geometry(
            src_x, src_y, src_z, rec_x, rec_y, rec_z, t0, tn, src_type, f0=f0
        )
        self.checkpointing = checkpointing
        self.kernel = kernel
        self.time_order = time_order
        self.karguments = {}
        self._dswap_opt = {
            "dswap": dswap,
            "dswap_disks": dswap_disks,
            "dswap_folder": dswap_folder,
            "dswap_folder_path": dswap_folder_path,
            "dswap_compression": dswap_compression,
            "dswap_compression_value": dswap_compression_value,
        }

        super().__init__(
            dtype=np.dtype(dtype),
            dims=vp.shape,
            dimsd=(len(src_x), len(rec_x), self.geometry.nt),
            explicit=False,
            name=name,
        )
        self._register_multiplications(op_name)

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        qp: NDArray,
        b: NDArray,
        space_order: int = 6,
        nbl: int = 20,
        dt: int = None,
    ) -> None:
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in m/s
        qp : :obj:`numpy.ndarray
            P-wave attenuation
        b : :obj:`numpy.ndarray
            Buoyancy
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries

        """
        self.space_order = space_order
        self.model = Model(
            space_order=space_order,
            vp=vp * 1e-3,
            qp=qp,
            b=b,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            dt=dt,
        )

    def _create_geometry(
        self,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str,
        f0: float = 20.0,
    ) -> None:
        """Create geometry and time axis

        Parameters
        ----------
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in m
        src_y : :obj:`numpy.ndarray`
            Source y-coordinates in m
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in m
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in m
        rec_y : :obj:`numpy.ndarray`
            Receiver y-coordinates in m
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in m
        t0 : :obj:`float`
            Initial time
        tn : :obj:`int`
            Number of time samples
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, self.model.dim))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, -1] = src_z
        if self.model.dim == 3:
            src_coordinates[:, 1] = src_y

        rec_coordinates = np.empty((nrec, self.model.dim))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, -1] = rec_z
        if self.model.dim == 3:
            rec_coordinates[:, 1] = rec_y

        self.geometry = AcquisitionGeometry(
            self.model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 * 1e-3,
        )

    def _fwd_oneshot(self, solver: AcousticWaveSolver, v: NDArray) -> NDArray:
        """Forward modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """
        d = solver.forward(**self.karguments)[0]
        d = d.resample(solver.geometry.dt).data[:][: solver.geometry.nt].T
        return d

    def _fwd_allshots(self, v: NDArray) -> NDArray:
        """Forward modelling for all shots

        Parameters
        -----------
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        # solve
        solver = ViscoacousticWaveSolver(
            self.model,
            geometry,
            space_order=self.space_order,
            kernel=self.kernel,
            time_order=self.time_order,
            **self._dswap_opt,
        )
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._fwd_oneshot(solver, v)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _adj_allshots(self, v: NDArray) -> NDArray:
        raise Exception("Method not yet implemented")

    def _register_multiplications(self, op_name: str) -> None:
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
            self._acoustic_rmatvec = self._adj_allshots

    def create_receiver(
        self, name, rx=None, ry=None, rz=None, t0=None, tn=None, dt=None
    ):

        if self.model.dim == 2 and ry is not None:
            raise Exception("Attempting to create 3D receiver for a 2D operator!")

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt

        rx = rx if rx is not None else self.geometry.rec_positions[:, 0]
        rz = rz if rz is not None else self.geometry.rec_positions[:, -1]
        if self.model.dim == 3:
            ry = ry if ry is not None else self.geometry.rec_positions[:, 1]

        nrec = len(rx)

        rec_coordinates = np.empty((nrec, 3))
        rec_coordinates[:, 0] = rx
        rec_coordinates[:, -1] = rz
        if self.model.dim == 3:
            rec_coordinates[:, 1] = ry

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)
        return Receiver(
            name=name,
            grid=self.geometry.grid,
            time_range=time_axis,
            npoint=nrec,
            coordinates=rec_coordinates,
        )

    def create_source(
        self,
        name,
        sx=None,
        sy=None,
        sz=None,
        t0=None,
        tn=None,
        dt=None,
        f0=None,
        src_type=None,
    ):

        if self.model.dim == 2 and sy is not None:
            raise Exception("Attempting to create 3D source for a 2D operator!")

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt
        f0 = f0 or self.geometry.f0

        src_type = src_type or self.geometry.src_type

        sx = sx or self.geometry.src_positions[:, 0]
        sz = sz or self.geometry.src_positions[:, -1]
        if self.model.dim == 3:
            sy = sy or self.geometry.src_positions[:, 1]

        nsrc = len(sx)

        src_coordinates = np.empty((nsrc, 3))
        src_coordinates[:, 0] = sx
        src_coordinates[:, -1] = sz
        if self.model.dim == 3:
            src_coordinates[:, 1] = sy

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)

        return sources[src_type](
            name=name,
            grid=self.geometry.grid,
            f0=f0,
            time_range=time_axis,
            npoint=nsrc,
            coordinates=src_coordinates,
            t0=t0,
        )

    def add_args(self, **kwargs):
        self.karguments = kwargs

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_matvec(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_rmatvec(x)
        return y


class ViscoAcousticWave2D(_ViscoAcousticWave):
    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        qp: NDArray,
        b: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        kernel: str = "sls",
        time_order: int = 2,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        op_name: str = "fwd",
        dt: int = None,
        **kwargs,
    ) -> None:

        if len(shape) != 2:
            raise Exception(
                "Attempting to create a 3D operator using a 2D intended class!"
            )

        super().__init__(
            shape=shape,
            origin=origin,
            spacing=spacing,
            vp=vp,
            qp=qp,
            b=b,
            src_x=src_x,
            src_z=src_z,
            rec_x=rec_x,
            rec_z=rec_z,
            t0=t0,
            tn=tn,
            src_type=src_type,
            space_order=space_order,
            kernel=kernel,
            time_order=time_order,
            nbl=nbl,
            f0=f0,
            checkpointing=checkpointing,
            dtype=dtype,
            name=name,
            op_name=op_name,
            dt=dt,
            **kwargs,
        )

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl]


class ViscoAcousticWave3D(_ViscoAcousticWave):
    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        qp: NDArray,
        b: NDArray,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        kernel: str = "sls",
        time_order: int = 2,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        op_name: str = "fwd",
        dt: int = None,
        **kwargs,
    ) -> None:

        if len(shape) != 3:
            raise Exception(
                "Attempting to create a 2D operator with a 3D intended class!"
            )

        super().__init__(
            shape=shape,
            origin=origin,
            spacing=spacing,
            vp=vp,
            qp=qp,
            b=b,
            src_x=src_x,
            src_y=src_y,
            src_z=src_z,
            rec_x=rec_x,
            rec_y=rec_y,
            rec_z=rec_z,
            t0=t0,
            tn=tn,
            src_type=src_type,
            space_order=space_order,
            kernel=kernel,
            time_order=time_order,
            nbl=nbl,
            f0=f0,
            checkpointing=checkpointing,
            dtype=dtype,
            name=name,
            op_name=op_name,
            dt=dt,
            **kwargs,
        )

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl, nbl:-nbl]
