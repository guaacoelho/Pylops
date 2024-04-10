__all__ = ["AcousticWave2D", "AcousticWave3D", "ElasticWave2D", "ElasticWave3D"]

from typing import Tuple

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike

devito_message = deps.devito_import("the twoway module")

if devito_message is None:
    from devito.builtins import initialize_function

    from examples.seismic import AcquisitionGeometry, Model, Receiver
    from examples.seismic.acoustic import AcousticWaveSolver
    from examples.seismic.stiffness import ISOSeismicModel, IsoElasticWaveSolver
    from examples.seismic.source import TimeAxis
    from examples.seismic.utils import sources


class AcousticWave2D(LinearOperator):
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
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        # create model
        self._create_model(shape, origin, spacing, vp, space_order, nbl)
        self._create_geometry(src_x, src_z, rec_x, rec_z, t0, tn, src_type, f0=f0)
        self.checkpointing = checkpointing

        super().__init__(
            dtype=np.dtype(dtype),
            dims=vp.shape,
            dimsd=(len(src_x), len(rec_x), self.geometry.nt),
            explicit=False,
            name=name,
        )
        self._register_multiplications(op_name)

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl]

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        space_order: int = 6,
        nbl: int = 20,
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
        )

    def _create_geometry(
        self,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
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
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, 2))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_z

        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_z

        self.geometry = AcquisitionGeometry(
            self.model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 * 1e-3,
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
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)

        # source wavefield
        u0 = solver.forward(save=True)[1]
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

    def _born_oneshot(self, isrc: int, dm: NDArray) -> NDArray:
        """Born modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        dm : :obj:`np.ndarray`
            Model perturbation

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

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

        # set perturbation
        dmext = np.zeros(self.model.grid.shape, dtype=np.float32)
        dmext[self.model.nbl : -self.model.nbl, self.model.nbl : -self.model.nbl] = dm

        # solve
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)
        d = solver.jacobian(dmext)[0]
        d = d.resample(geometry.dt).data[:][: geometry.nt].T
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
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            d = self._born_oneshot(isrc, dm)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _bornadj_oneshot(self, isrc, dobs):
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
        # create boundary data
        recs = self.geometry.rec.copy()
        recs.data[:] = dobs.T[:]

        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)

        # source wavefield
        if hasattr(self, "src_wavefield"):
            u0 = self.src_wavefield[isrc]
        else:
            u0 = solver.forward(save=True)[1]
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
        nsrc = self.geometry.src_positions.shape[0]
        mtot = np.zeros(self.model.shape, dtype=np.float32)

        for isrc in range(nsrc):
            m = self._bornadj_oneshot(isrc, dobs[isrc])
            mtot += self._crop_model(m.data, self.model.nbl)
        return mtot

    def _fwd_oneshot(self, isrc: int, v: NDArray) -> NDArray:
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

        # Update model.vp using data received as a parameter
        initialize_function(self.model.vp, v * 1e-3, self.model.padsizes)

        # solve
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)
        d = solver.forward()[0]
        d = d.resample(geometry.dt).data[:][: geometry.nt].T
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
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            d = self._fwd_oneshot(isrc, v)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _register_multiplications(self, op_name: str) -> None:
        if op_name == "born":
            self._acoustic_matvec = self._born_allshots
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
        self._acoustic_rmatvec = self._bornadj_allshots

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_matvec(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_rmatvec(x)
        return y


class AcousticWave3D(LinearOperator):
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
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        # create model
        self._create_model(shape, origin, spacing, vp, space_order, nbl)
        self._create_geometry(
            src_x, src_y, src_z, rec_x, rec_y, rec_z, t0, tn, src_type, f0=f0
        )
        self.checkpointing = checkpointing

        super().__init__(
            dtype=np.dtype(dtype),
            dims=vp.shape,
            dimsd=(len(src_x), len(rec_x), self.geometry.nt),
            explicit=False,
            name=name,
        )
        self._register_multiplications(op_name)

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl, nbl:-nbl]

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        space_order: int = 6,
        nbl: int = 20,
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
        src_coordinates = np.empty((nsrc, 3))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_y
        src_coordinates[:, 2] = src_z

        rec_coordinates = np.empty((nrec, 3))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_y
        rec_coordinates[:, 2] = rec_z

        self.geometry = AcquisitionGeometry(
            self.model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 * 1e-3,
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
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)

        # source wavefield
        u0 = solver.forward(save=True)[1]
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

    def _born_oneshot(self, isrc: int, dm: NDArray) -> NDArray:
        """Born modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        dm : :obj:`np.ndarray`
            Model perturbation

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

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

        # set perturbation
        dmext = np.zeros(self.model.grid.shape, dtype=np.float32)
        dmext[
            self.model.nbl : -self.model.nbl,
            self.model.nbl : -self.model.nbl,
            self.model.nbl : -self.model.nbl,
        ] = dm

        # solve
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)
        d = solver.jacobian(dmext)[0]
        d = d.resample(geometry.dt).data[:][: geometry.nt].T
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
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            d = self._born_oneshot(isrc, dm)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _bornadj_oneshot(self, isrc, dobs):
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
        # create boundary data
        recs = self.geometry.rec.copy()
        recs.data[:] = dobs.T[:]

        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)

        # source wavefield
        if hasattr(self, "src_wavefield"):
            u0 = self.src_wavefield[isrc]
        else:
            u0 = solver.forward(save=True)[1]
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
        nsrc = self.geometry.src_positions.shape[0]
        mtot = np.zeros(self.model.shape, dtype=np.float32)

        for isrc in range(nsrc):
            m = self._bornadj_oneshot(isrc, dobs[isrc])
            mtot += self._crop_model(m.data, self.model.nbl)
        return mtot

    def _fwd_oneshot(self, isrc: int, v: NDArray) -> NDArray:
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

        # Update model.vp using data received as a parameter
        initialize_function(self.model.vp, v * 1e-3, self.model.padsizes)

        # solve
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)
        d = solver.forward()[0]
        d = d.resample(geometry.dt).data[:][: geometry.nt].T
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
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            d = self._fwd_oneshot(isrc, v)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _register_multiplications(self, op_name: str) -> None:
        if op_name == "born":
            self._acoustic_matvec = self._born_allshots
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
        self._acoustic_rmatvec = self._bornadj_allshots

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_matvec(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_rmatvec(x)
        return y


class ElasticWave2D(LinearOperator):
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
        op_name: str = "fwd",
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        # create model
        self._create_model(shape, origin, spacing, vp, vs, rho, space_order, nbl)
        self._create_geometry(src_x, src_z, rec_x, rec_z, t0, tn, src_type, f0=f0)
        self.checkpointing = checkpointing
        self.num_outs = 3
        self.karguments = {}
        super().__init__(
            dtype=np.dtype(dtype),
            dims=vp.shape,
            dimsd=(self.num_outs, len(src_x), len(rec_x), self.geometry.nt),
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
            vp=vp * 1e-3,
            vs=vs * 1e-3,
            rho=rho,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
        )

    def _create_geometry(
        self,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
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
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, 2))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_z

        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_z

        self.geometry = AcquisitionGeometry(
            self.model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 * 1e-3,
        )

    def _fwd_oneshot(self, isrc: int, v: NDArray) -> NDArray:
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

        # Update model.vp using data received as a parameter
        initialize_function(self.model.vp, v * 1e-3, self.model.padsizes)

        # solve
        solver = IsoElasticWaveSolver(self.model, geometry, space_order=self.space_order)
        rec_data = list(solver.forward(**self.karguments)[0:3])

        for ii, d in enumerate(rec_data):
            rec_data[ii] = d.resample(geometry.dt).data[:][: geometry.nt].T
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
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            d = self._fwd_oneshot(isrc, v)
            dtot.append(d)

        # Adjust dimensions
        rec_data = list(zip(*dtot))

        return np.array(rec_data)

    def _adj_allshots(self, v: NDArray) -> NDArray:
        raise Exception("Method not yet implemented")

    def _register_multiplications(self, op_name: str) -> None:
        self.op_name = op_name
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
            self._acoustic_rmatvec = self._adj_allshots
        else:
            raise Exception("The operator's name '%s' is not valid." % op_name)

    def create_receiver(self, name, rx=None, rz=None, t0=None, tn=None, dt=None):

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt

        rx = rx if rx is not None else self.geometry.rec_positions[:, 0]
        rz = rz if rz is not None else self.geometry.rec_positions[:, 1]

        nrec = len(rx)

        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = rx
        rec_coordinates[:, 1] = rz

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)
        return Receiver(name=name, grid=self.geometry.grid,
                        time_range=time_axis, npoint=nrec,
                        coordinates=rec_coordinates)

    def create_source(self, name, sx=None, sz=None, t0=None, tn=None, dt=None, f0=None, src_type=None):

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt
        f0 = f0 or self.geometry.f0

        src_type = src_type or self.geometry.src_type

        sx = sx or self.geometry.src_positions[:, 0]
        sz = sz or self.geometry.src_positions[:, 1]

        nsrc = len(sx)

        src_coordinates = np.empty((nsrc, 2))
        src_coordinates[:, 0] = sx
        src_coordinates[:, 1] = sz

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)

        return sources[src_type](name=name, grid=self.geometry.grid, f0=f0,
                                 time_range=time_axis, npoint=nsrc,
                                 coordinates=src_coordinates, t0=t0)

    def add_args(self, **kwargs):
        self.karguments = kwargs

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


class ElasticWave3D(LinearOperator):
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
        op_name: str = "fwd",
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        # create model
        self._create_model(shape, origin, spacing, vp, vs, rho, space_order, nbl)
        self._create_geometry(src_x, src_y, src_z, rec_x, rec_y, rec_z, t0, tn, src_type, f0=f0)
        self.checkpointing = checkpointing
        self.num_outs = 4
        self.karguments = {}
        super().__init__(
            dtype=np.dtype(dtype),
            dims=vp.shape,
            dimsd=(self.num_outs, len(src_x), len(rec_x), self.geometry.nt),
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
            vp=vp * 1e-3,
            vs=vs * 1e-3,
            rho=rho,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
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
        src_coordinates = np.empty((nsrc, 3))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_y
        src_coordinates[:, -1] = src_z

        rec_coordinates = np.empty((nrec, 3))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_y
        rec_coordinates[:, -1] = rec_z

        self.geometry = AcquisitionGeometry(
            self.model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 * 1e-3,
        )

    def _fwd_oneshot(self, isrc: int, v: NDArray) -> NDArray:
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

        # Update model.vp using data received as a parameter
        initialize_function(self.model.vp, v * 1e-3, self.model.padsizes)

        # solve
        solver = IsoElasticWaveSolver(self.model, geometry, space_order=self.space_order)
        rec_data = list(solver.forward(**self.karguments)[0:4])

        for ii, d in enumerate(rec_data):
            rec_data[ii] = d.resample(geometry.dt).data[:][: geometry.nt].T
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
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            d = self._fwd_oneshot(isrc, v)
            dtot.append(d)

        # Adjust dimensions
        rec_data = list(zip(*dtot))

        return np.array(rec_data)

    def _adj_allshots(self, v: NDArray) -> NDArray:
        raise Exception("Method not yet implemented")

    def _register_multiplications(self, op_name: str) -> None:
        self.op_name = op_name
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
            self._acoustic_rmatvec = self._adj_allshots
        else:
            raise Exception("The operator's name '%s' is not valid." % op_name)

    def create_receiver(self, name, rx=None, ry=None, rz=None, t0=None, tn=None, dt=None):

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt

        rx = rx if rx is not None else self.geometry.rec_positions[:, 0]
        ry = ry if ry is not None else self.geometry.rec_positions[:, 1]
        rz = rz if rz is not None else self.geometry.rec_positions[:, -1]

        nrec = len(rx)

        rec_coordinates = np.empty((nrec, 3))
        rec_coordinates[:, 0] = rx
        rec_coordinates[:, 1] = ry
        rec_coordinates[:, -1] = rz

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)
        return Receiver(name=name, grid=self.geometry.grid,
                        time_range=time_axis, npoint=nrec,
                        coordinates=rec_coordinates)

    def create_source(self, name, sx=None, sy=None, sz=None, t0=None, tn=None, dt=None, f0=None, src_type=None):

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt
        f0 = f0 or self.geometry.f0

        src_type = src_type or self.geometry.src_type

        sx = sx or self.geometry.src_positions[:, 0]
        sy = sy or self.geometry.src_positions[:, 1]
        sz = sz or self.geometry.src_positions[:, -1]

        nsrc = len(sx)

        src_coordinates = np.empty((nsrc, 3))
        src_coordinates[:, 0] = sx
        src_coordinates[:, 1] = sy
        src_coordinates[:, -1] = sz

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)

        return sources[src_type](name=name, grid=self.geometry.grid, f0=f0,
                                 time_range=time_axis, npoint=nsrc,
                                 coordinates=src_coordinates, t0=t0)

    def add_args(self, **kwargs):
        self.karguments = kwargs

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
