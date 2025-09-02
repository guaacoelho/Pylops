import logging
from copy import deepcopy
from typing import TypeVar, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike

from .wave import _Wave

devito_message = deps.devito_import("the twoway module")

if devito_message is None:
    from devito import (
        DiskSwapConfig,
        Eq,
        Function,
        Operator,
        TensorTimeFunction,
        VectorFunction,
        VectorTimeFunction,
    )
    from devito.builtins import initialize_function

    from examples.seismic import AcquisitionGeometry
    from examples.seismic.stiffness import (
        ElasticModel,
        GenericElasticWaveSolver,
        elastic_stencil,
    )
    from examples.seismic.utils import PointSource

MPIComm = TypeVar("mpi4py.MPI.Comm")


class _ElasticWave(_Wave):
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
        save_wavefield: bool = False,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_path: str = None,
        dswap_compression: str = None,
        dswap_compression_value: float | int = None,
        dswap_verbose: bool = False,
        **kwargs,
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        init_logger = logging.getLogger("init_logger")
        init_logger.setLevel(logging.WARNING)

        is_2d = len(shape) == 2
        is_3d = len(shape) == 3

        if is_2d and (rec_y is not None or src_y is not None):
            raise Exception(
                "Attempting to create a 3D operator using a 2D intended class!"
            )

        if is_3d and (rec_y is None or src_y is None):
            raise Exception(
                "Attempting to create a 2D operator using a 3D intended class!"
            )

        # create model
        self._create_model(
            shape, origin, spacing, vp, vs, rho, space_order, nbl, dt, **kwargs
        )
        self._create_geometry(
            src_x, src_y, src_z, rec_x, rec_y, rec_z, t0, tn, src_type, f0=f0
        )
        self.checkpointing = checkpointing
        self.par = par
        self.karguments = {}
        dim = self.model.dim

        if dswap and save_wavefield:
            init_logger.warning(
                "Disk swap is incompatible with wave saving. Disabling wave saving"
            )

        self.save_wavefield = save_wavefield if (not dswap) else False
        if self.save_wavefield:
            self.src_wavefield = []

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

        self._dswap_opt = {
            "dswap": dswap,
            "dswap_disks": dswap_disks,
            "dswap_folder": dswap_folder,
            "dswap_path": dswap_path,
            "dswap_compression": dswap_compression,
            "dswap_compression_value": dswap_compression_value,
            "dswap_verbose": dswap_verbose,
        }

        self._register_multiplications(op_name)

    def _crop_stencil(self, m):
        """Remove absorbing boundaries from model"""
        nbl = self.model.nbl
        cropped_stencils = []
        for stencil in m:
            slices = tuple(slice(nbl, -nbl) for _ in range(stencil.ndim))
            cropped_stencils.append(stencil[slices])
        return np.array(cropped_stencils)

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
        **kwargs,
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
        parameters_list = [
            "epsilon",
            "delta",
            "phi",
            "gamma",
            "C11",
            "C22",
            "C33",
            "C44",
            "C55",
            "C66",
            "C12",
            "C21",
            "C13",
            "C31",
            "C23",
            "C32",
        ]

        physical_parameters = {
            arg: value for arg, value in kwargs.items() if arg in parameters_list
        }

        self.space_order = space_order
        self.model = ElasticModel(
            space_order=space_order,
            vp=vp / 1000,
            vs=vs / 1000,
            rho=rho,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            bcs="damp",
            nbl=nbl,
            dt=dt,
            **physical_parameters,
        )

    def _fwd_oneshot(self, solver: GenericElasticWaveSolver, v: NDArray) -> NDArray:
        """Forward modelling for one shot

        Parameters
        ----------
        solver : :obj:`GenericElasticWaveSolver`
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

        *rec_data, v = solver.forward(**self.karguments, save=self.save_wavefield)[
            0 : dim + 2
        ]
        if self.save_wavefield:
            self.src_wavefield.append(v)

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
        solver = GenericElasticWaveSolver(
            self.model, geometry, space_order=self.space_order
        )

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._fwd_oneshot(solver, v)
            dtot.append(deepcopy(d))

        # Adjust dimensions
        rec_data = list(zip(*dtot))

        return np.array(rec_data)

    def _imaging_operator(self, img: VectorFunction) -> Operator:
        """Imaging operator built using Devito

        Parameters
        ----------
        model : :obj:`Model`
            Devito velocity model
        img : :obj:`VectorFunction`
            Image function
        geometry : :obj:`Geometry`
            Devito geometry
        space_order : :obj:`int`
            Spatial ordering of FD stencil
        dt_ref : :obj:`float`
            Time discretization step
        Returns
        -------
        imop : :obj:`Operator`
            The imaging operator
        u : :obj:`VectorTimeFunction`
            Backward wavefield

        """
        # Define the wavefield with the size of the model and the time dimension
        dswap = self._dswap_opt["dswap"]
        model = self.model
        geometry = self.geometry
        space_order = self.model.space_order
        dt_ref = self.geometry.dt

        v = VectorTimeFunction(
            name="v",
            grid=model.grid,
            save=geometry.nt if not dswap else None,
            space_order=space_order,
            time_order=1,
        )

        u = VectorTimeFunction(
            name="u", grid=model.grid, space_order=space_order, time_order=1
        )
        sig = TensorTimeFunction(
            name="sig", grid=model.grid, space_order=space_order, time_order=1
        )

        eqn = elastic_stencil(
            model, u, sig, forward=False, par=self.karguments.get("par", self.par)
        )

        dt = dt_ref
        b = 1.0 / model.rho

        # Define residual injection at the location of the forward receivers
        rec_vx = PointSource(
            name="rec_vx",
            grid=model.grid,
            time_range=geometry.time_axis,
            coordinates=geometry.rec_positions,
        )

        rec_vz = PointSource(
            name="rec_vz",
            grid=model.grid,
            time_range=geometry.time_axis,
            coordinates=geometry.rec_positions,
        )

        rec_vy = PointSource(
            name="rec_vy",
            grid=model.grid,
            time_range=geometry.time_axis,
            coordinates=geometry.rec_positions,
        )

        rec_term_vx = rec_vx.inject(field=u[0].backward, expr=dt * rec_vx * b)
        rec_term_vz = rec_vz.inject(field=u[-1].backward, expr=dt * rec_vz * b)

        rec_expr = rec_term_vx + rec_term_vz

        if model.grid.dim == 3:
            rec_expr += rec_vy.inject(field=u[1].backward, expr=dt * rec_vy * b)

        ixx_update = [Eq(img[0], img[0] + v[0] * u[0])]
        izz_update = [Eq(img[-1], img[-1] + v[-1] * u[-1])]

        img_update = ixx_update + izz_update

        if model.grid.dim == 3:
            img_update += [Eq(img[1], img[1] + v[1] * u[1])]

        opt = {}
        if dswap:
            dconfig = DiskSwapConfig(
                functions=[v],
                mode="read",
                path=self._dswap_opt["dswap_path"],
                folder=self._dswap_opt["dswap_folder"],
                verbose=self._dswap_opt["dswap_verbose"],
            )

            opt.update({"opt": ("advanced", {"disk-swap": dconfig})})

        return Operator(
            eqn + rec_expr + img_update, subs=model.spacing_map, name="Imaging", **opt
        )

    def _imaging_oneshot(
        self,
        isrc: int,
        recs: NDArray,
        imaging: Operator,
        solver: GenericElasticWaveSolver,
        **kwargs,
    ) -> None:
        """Imaging modelling for one shot

        Parameters
        ----------
        isrc : :obj:`float`
            Index of source to model
        recs : :obj:`np.ndarray`
            Receiver observed data to inject
        imaging : :obj:`Operator`
            Imaging operator build with Devito
        solver : :obj:`GenericElasticWaveSolver`
            Devito's solver object

        Returns
        -------

        """
        vfields = kwargs.copy()
        dim = self.model.dim

        rec_vx = PointSource(
            name="dobs_vx_resam",
            grid=self.model.grid,
            time_range=self.geometry.time_axis,
            coordinates=self.geometry.rec_positions,
            data=recs[0].T,
        )
        rec_vz = PointSource(
            name="dobs_vz_resam",
            grid=self.model.grid,
            time_range=self.geometry.time_axis,
            coordinates=self.geometry.rec_positions,
            data=recs[-1].T,
        )

        vfields.update({"rec_vx": rec_vx, "rec_vz": rec_vz})

        if dim == 3:
            rec_vy = PointSource(
                name="dobs_vy_resam",
                grid=self.model.grid,
                time_range=self.geometry.time_axis,
                coordinates=self.geometry.rec_positions,
                data=recs[1].T,
            )

            vfields.update({"rec_vy": rec_vy})

        if solver:
            v0 = solver.forward(
                par=self.karguments.get("par", self.par),
                save=True if not self._dswap_opt["dswap"] else False,
            )[dim + 1]
        else:
            v0 = self.src_wavefield[isrc]

        vfields.update({k.name: k for k in v0})
        vfields.update({"dt": self.model.critical_dt})
        imaging(**vfields)

    def _imaging_allshots(self, dobs: NDArray, **kwargs) -> NDArray:
        """Imaging modelling for all shots

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
        imf : :obj:`np.ndarray`
            Image generated by all shots

        """
        if hasattr(self, "src_wavefield") and self.src_wavefield:
            solver = None
        else:
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

            solver = GenericElasticWaveSolver(
                self.model, geometry, space_order=self.space_order, **self._dswap_opt
            )

        image = VectorFunction(name="image", grid=self.model.grid)

        nsrc = self.geometry.src_positions.shape[0]
        for isrc in range(nsrc):
            imaging = self._imaging_operator(image, **kwargs)
            # For each dobs get data equivalent to isrc shot
            rec_i = [rec[isrc] for rec in dobs]

            # Positioning forward propagation src, if needed
            if solver:
                solver.geometry.src_positions = self.geometry.src_positions[isrc, :]

            self._imaging_oneshot(isrc, rec_i, imaging, solver)

        shape = self.model.grid.shape
        ndim = len(shape)
        imf = np.zeros((ndim, *shape), dtype=np.float32)
        for ii, im in enumerate(image):
            imf[ii] = im.data

        return imf

    def rtm(self, recs: NDArray, **kwargs) -> NDArray:
        controller = kwargs.pop("mpi_controller", None)
        image = self._imaging_allshots(recs, **kwargs)
        cropped_image = self._crop_stencil(image)

        if controller:
            return controller.build_result(cropped_image)
        else:
            return cropped_image

    def _grad_oneshot(self, isrc, dobs, solver: GenericElasticWaveSolver):
        """Adjoint gradient modelling for one shot

        Parameters
        ----------
        isrc : :obj:`float`
            Index of source to model
        dobs : :obj:`np.ndarray`
            Observed data to inject
        solver : :obj:`GenericElasticWaveSolver`
            Devito's solver object

        Returns
        -------
        model : :obj:`np.ndarray`
            Model

        """
        # set disk_swap bool
        dswap = self._dswap_opt.get("dswap", False)

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

        # If "par" was not passed as a parameter to forward execution, use the operator's default value
        self.karguments["par"] = self.karguments.get("par", self.par)

        # source wavefield
        if hasattr(self, "src_wavefield"):
            u0 = self.src_wavefield[isrc]
        else:
            par = self.karguments.get("par")
            u0 = solver.forward(save=True if not dswap else False, par=par)[dim + 1]

        # adjoint modelling (reverse wavefield)
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

        solver = GenericElasticWaveSolver(
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
