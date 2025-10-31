__all__ = [
    "AcousticWave2D",
    "AcousticWave3D",
    "ElasticWave2D",
    "ElasticWave3D",
    "ViscoAcousticWave2D",
    "ViscoAcousticWave3D",
    "MultiparameterViscoAcoustic2D",
    "MultiparameterViscoAcoustic3D",
]

import numpy as np
from scipy import interpolate

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.typing import NDArray
from pylops.waveeqprocessing._propertiesmixin import PhysicalPropertiesMixin

devito_message = deps.devito_import("the twoway module")

if devito_message is None:
    from examples.seismic import AcquisitionGeometry, Receiver
    from examples.seismic.source import TimeAxis
    from examples.seismic.utils import PointSource, sources


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


class _Wave(LinearOperator, PhysicalPropertiesMixin):
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

    def updatesrc(self, wav, method="padding"):
        """Update source wavelet

        This routine allows users to pass a custom source
        wavelet to replace the source wavelet generated during
        the object's initialization.

        Parameters
        ----------
        wav : :obj:`numpy.ndarray`
            Wavelet
        method : :str
            Method representing how the data will be filled up to the total nt.
            - padding
            - resample

        """
        wav = wav.reshape(-1)
        if method not in {"padding", "resample"}:
            raise ValueError(f"Invalid method '{method}'. Supported methods are 'padding', 'resample'.")

        if method == "padding":
            wav_data = np.pad(wav, (0, self.geometry.nt - len(wav)))
        elif method == "resample":
            wav_data = self.resample(wav, self.geometry.nt)

        self.wav = _CustomSource(
            name="src",
            grid=self.model.grid,
            wav=wav_data,
            time_range=self.geometry.time_axis,
        )

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

    def _update_dimensions(self, new_dims, new_dimsd):
        """
        Update the dimensions and shape of the object.

        Parameters:
        -----------
        new_dims : tuple
            The new value of dims.
        new_dimsd : tuple
            The new value of dimsd.

        """

        del self.dims
        del self.dimsd

        self.shape = (np.prod(new_dimsd), np.prod(new_dims))

        self.dims = new_dims
        self.dimsd = new_dimsd

    def _update_geometry(self, rx, rz, sx, sz, nrecs):
        """
        Update the geometry with new receiver and source positions.

        Parameters
        ----------
        rx : array-like
            Array containing the x-coordinates of the receivers.
        rz : array-like
            Array containing the z-coordinates of the receivers.
        sx : float
            x-coordinate of the source.
        sz : float
            z-coordinate of the source.
        nrecs : int
            Number of receivers.
        tn : float
            Final recording time.

        Notes
        -----
        This method updates the `geometry` attribute of the object by creating
        a new `AcquisitionGeometry` instance with the provided receiver and
        source positions, as well as the updated final recording time.

        For now, it only works for 2D operatores.
        """

        new_rec_positions = np.zeros((nrecs, 2))
        new_rec_positions[:, 0] = rx
        new_rec_positions[:, -1] = rz

        new_src_positions = np.zeros((1, 2))
        new_src_positions[:, 0] = sx
        new_src_positions[:, -1] = sz

        self.geometry = AcquisitionGeometry(
            self.model,
            new_rec_positions,
            new_src_positions,
            self.geometry.t0,
            self.geometry.tn,
            src_type=self.geometry.src_type,
            f0=self.geometry.f0,
        )

    def _update_op_coords(self, id_src, relative_coords=False):
        """
        Update operator coordinates and dimensions based on SEGY file data.

        This method retrieves the source and receiver coordinates, time sampling
        information, and other relevant parameters from the SEGY file associated
        with the operator. It updates the internal model time step, geometry, and
        dimensions as needed.

        Raises:
            Exception: If the SEGY file is used but the shot index (`id_src`) is not defined.
        """
        if relative_coords:
            src_coords, rec_coords = self.segyReader.getRelativeCoords(id_src)
        else:
            src_coords, rec_coords = self.segyReader.getCoords(id_src)
        rx, rz = rec_coords
        sx, sz = src_coords

        nrec = len(rx)

        # Check if the number of receivers is variable and differs from the current geometry.
        # If so, update the dimensions to match the new number of receivers for the current shot.
        if nrec != self.geometry.nrec:
            self._update_dimensions(
                new_dims=self.dims, new_dimsd=(1, nrec, self.geometry.nt)
            )

        self._update_geometry(rx, rz, sx, sz, nrec)

    def resample(self, data, num):
        """
        Resample the input data to a new number of time steps.

        This method determines whether the input data corresponds to receiver data
        or source data based on its shape and calls the appropriate resampling method.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            Input data to be resampled. Can be either source or receiver data.
        num : :obj:`int`
            Number of time steps for the resampled data.
        """
        if len(data.shape) == 3:
            # receivers has shape (nshots, nrec, nt)
            return self._resample_rec(data, num)
        else:
            return self._resample_src(data, num)

    def _resample_src(self, data, num):
        nsteps = data.shape[0]

        time_range = TimeAxis(start=self.geometry.time_axis.start, stop=self.geometry.time_axis.stop, num=nsteps)
        new_data = np.zeros((num), dtype=np.float32)

        src = PointSource(name='src', grid=self.model.grid, npoint=1, time_range=time_range)
        src.data[:] = data[:].reshape(-1, 1)
        src = src.resample(num=num)

        new_data[:] = src.data.T
        return new_data

    def _resample_rec(self, data, num):

        nshots, ntraces, nsteps = data.shape
        time_range = TimeAxis(start=self.geometry.time_axis.start, stop=self.geometry.time_axis.stop, num=nsteps)

        new_time_range = TimeAxis(start=self.geometry.time_axis.start, stop=self.geometry.time_axis.stop, num=num)
        new_traces = np.zeros((nshots, ntraces, num), dtype=np.float32)
        for shot_id in range(nshots):
            for i in range(ntraces):
                tck = interpolate.splrep(time_range.time_values,
                                         data[shot_id, i, :], k=3)
                new_traces[shot_id, i] = interpolate.splev(new_time_range.time_values, tck)

        return new_traces

    def add_args(self, **kwargs):
        self.karguments = kwargs

    def update_args(self, **kwargs):
        self.karguments.update(kwargs)

    def set_shotID(self, id_src, relative_coords=False):
        """
        Set the ID for the shot that will be executed by the operator.

        This method updates the operator's internal state to process a specific
            raise Exception("Can not set shot ID for an operator that doesn't have segyReader")
        for the given shot ID and updates the geometry accordingly.

        Parameters
        ----------
        id_src : int
            The ID of the shot to be executed. This corresponds to the shot index
            in the SEGY file.
        relative_coords : bool, optional
            If True, the coordinates will be treated as relative to the model's
            origin. If False, the coordinates are treated as absolute. Default is False.

        Raises
        ------
        Exception
            If the operator does not have a SEGY reader or if the shot ID is invalid.
        """
        segyReader = getattr(self, "segyReader", None)

        if not segyReader:
            raise Exception(
                "Can not set shot ID for a operator that doesn't have segyReader"
            )

        self._update_op_coords(id_src, relative_coords=relative_coords)

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        if len(m.shape) == 2:
            return m[nbl:-nbl, nbl:-nbl]
        else:
            return m[nbl:-nbl, nbl:-nbl, nbl:-nbl]


from .waveequations.acousticwave import AcousticWave2D, AcousticWave3D
from .waveequations.elasticwave import ElasticWave2D, ElasticWave3D
from .waveequations.viscoacousticwave import (
    MultiparameterViscoAcoustic2D,
    MultiparameterViscoAcoustic3D,
    ViscoAcousticWave2D,
    ViscoAcousticWave3D,
)
