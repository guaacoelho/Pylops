import segyio

import numpy as np
from pylops.basicoperators.vstack import VStack
from pylops.waveeqprocessing.twoway import AcousticWave2D
# from pylops.waveeqprocessing.twoway import  AcousticWave2D


__all__ = ['ReadSEGY2D']


class ReadSEGY2D():

    def __init__(self, segy_path):

        self.segyfile = segy_path
        self.table = self.make_lookup_table(segy_path)
        self.isRecVariable = self._isRecVariable()
        self.nsrc = len(self.table)

    def _isRecVariable(self):
        """
        Verify if the number of receivers per shots is Regular.

        return True if all shots has the same number of receivers, and False otherwise
        """
        # get a set of values representing the number of traces per shot. In other words, the number os receivers per shot
        n_traces_per_shots = set(v["Num_Traces"] for v in self.table.values())

        # If the lenght of the set is 1, means that all the shots has the same number os receivers
        return len(n_traces_per_shots) != 1

    def make_lookup_table(self, sgy_file):
        '''
        Make a lookup of shots, where the keys are the shot record IDs being
        searched (looked up)

        Made by Oscar Mojica
        '''
        lookup_table = {}
        with segyio.open(sgy_file, ignore_geometry=True) as f:
            index = None
            pos_in_file = 0
            for header in f.header:
                if int(header[segyio.TraceField.SourceGroupScalar]) < 0:
                    scalco = abs(1. / header[segyio.TraceField.SourceGroupScalar])
                else:
                    scalco = header[segyio.TraceField.SourceGroupScalar]
                if int(header[segyio.TraceField.ElevationScalar]) < 0:
                    scalel = abs(1. / header[segyio.TraceField.ElevationScalar])
                else:
                    scalel = header[segyio.TraceField.ElevationScalar]
                # Check to see if we're in a new shot
                index = header[segyio.TraceField.FieldRecord]
                if index not in lookup_table.keys():
                    lookup_table[index] = {}
                    lookup_table[index]['filename'] = sgy_file
                    lookup_table[index]['Trace_Position'] = pos_in_file
                    lookup_table[index]['Num_Traces'] = 1
                    lookup_table[index]['Source'] = (header[segyio.TraceField.SourceX] * scalco, header[segyio.TraceField.SourceY] * scalel)
                    lookup_table[index]['Receivers'] = []
                else:  # Not in a new shot, so increase the number of traces in the shot by 1
                    lookup_table[index]['Num_Traces'] += 1
                lookup_table[index]['Receivers'].append((header[segyio.TraceField.GroupX] * scalco, header[segyio.TraceField.GroupY] * scalel))
                pos_in_file += 1

        return lookup_table

    def getVelocityModel(self, path):
        # Read velocity model
        f = segyio.open(path, iline=segyio.tracefield.TraceField.FieldRecord,
                        xline=segyio.tracefield.TraceField.CDP)

        xl, il, t = f.xlines, f.ilines, f.samples
        if len(il) != 1:
            dims = (len(xl), len(il), len(t))
        else:
            dims = (len(xl), len(t))

        vp = f.trace.raw[:].reshape(dims)
        return vp

    def getSourceCoords(self, index=0):
        src_coords = np.array(self.table[index]['Source'])
        sx = np.array([src_coords[0]])
        sz = np.array([src_coords[-1]])

        return sx, sz

    def getReceiverCoords(self, index=0):
        recs_coords = np.array(self.table[index]['Receivers'])
        rx = np.array([coord[0] for coord in recs_coords])
        rz = np.array([coord[-1] for coord in recs_coords])

        return rx, rz

    def getCoords(self, index=0):
        rec_coords = self.getReceiverCoords(index)
        src_coords = self.getSourceCoords(index)

        return src_coords, rec_coords

    def getTn(self):
        # f = segyio.open(self.segyfile, ignore_geometry=True)
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            num_samples = len(f.samples)
            samp_int = f.bin[segyio.BinField.Interval] / 1000

        return (num_samples - 1) * samp_int

    def getDt(self):
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            dt = f.bin[segyio.BinField.Interval]  # microseconds
        return dt / 1000  # return in miliseconds

    def getData(self, index: int, chunk_size=1):
        """
        Return the data from a specific index. The data returned is a 1d array to
        match the output from VStack

        Parameters
        ----------
        index : :obj:`int`
            Index of the first shot of the chunck
        chunk_size : :obj:`int`
            Size of the chunk. Delimits how many shots will have data returned
        """
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            retrieved_shot = []

            for nshot in range(index, index + chunk_size):
                position = self.table[nshot]['Trace_Position']
                traces_in_shot = self.table[nshot]['Num_Traces']
                shot_traces = f.trace[position:position + traces_in_shot]

                for trace in shot_traces:
                    retrieved_shot.append(trace)
        # concatena todos os dados em um np.array 1D
        return np.concatenate(retrieved_shot)

    def _getOperatorChunk(self, shape, origin, spacing, vp0, nbl, space_order, t0, tn, src_type, f0, dtype, chunk, src0_idx, nshots):
        """
        Create a list of VStacks, where each VStack contains a specific number of AcousticWave2D operators.

        Parameters
        ----------
        shape : tuple
            Shape of the computational grid.
        origin : tuple
            Origin of the computational grid.
        spacing : tuple
            Spacing of the computational grid.
        vp0 : ndarray
            Velocity model.
        nbl : int
            Number of boundary layers.
        space_order : int
            Space order of the finite-difference scheme.
        t0 : float
            Start time of the simulation.
        tn : float
            End time of the simulation.
        src_type : str
            Type of the source (e.g., 'Ricker').
        f0 : float
            Peak frequency of the source.
        dtype : type
            Data type of the simulation (e.g., np.float32).
        chunk : int
            Number of shots/operators each VStack will encompass.
        src0_idx : int
            Starting index of the shots.
        nshots : int or None
            Total number of shots to process. If None, all shots are processed.

        Returns
        -------
        Aops : list of VStack
            A list of VStacks, where each VStack contains a specific number of AcousticWave2D operators.
        """
        nsrc = nshots if nshots else self.nsrc

        # Ajusta o limite superior para não ultrapassar o número total de fontes
        if src0_idx >= self.nsrc :
            raise ValueError(f"src0_idx ({src0_idx}) is out of bounds. It must be less than the total number of sources ({self.nsrc}).")

        # Limit the loop to always stay within the maximum number of shots. This ensures that
        # if the starting index is too high and the sum with the desired number of sources exceeds
        # the maximum index in the file, the loop remains within bounds.
        end_idx = min(src0_idx + nsrc, self.nsrc)

        Aops = []
        for isrc in range(src0_idx, end_idx, chunk):
            ops = []

            # Limit the loop to ensure the index always stays within the maximum number of sources.
            # This handles cases where not all VStacks have the same number of operators.
            # This check ensures the loop stops when there are no more desired shots to process,
            # instead of iterating through the entire chunk range.
            chunk_end = min(isrc + chunk, end_idx)

            for index in range(isrc, chunk_end):
                sx, sz = self.getSourceCoords(index)
                rx, rz = self.getReceiverCoords(index)

                Aop = AcousticWave2D(shape=shape, origin=origin, spacing=spacing, vp=vp0, nbl=nbl, space_order=space_order,
                                     src_x=sx, src_z=sz, rec_x=rx, rec_z=rz, t0=t0, tn=tn, src_type=src_type, f0=f0, dtype=dtype,
                                     op_name="fwd")
                ops.append(Aop)
            Aops.append(VStack(ops))
        return Aops

    def _getOperatorUnique(self, shape, origin, spacing, vp0, nbl, space_order, t0, tn, src_type, f0, dtype, src0_idx, nshots):
        nsrc = nshots if nshots else self.nsrc

        # Ajusta o limite superior para não ultrapassar o número total de fontes
        end_idx = min(src0_idx + nsrc, self.nsrc)

        Aops = []
        for isrc in range(src0_idx, end_idx):
            sx, sz = self.getSourceCoords(isrc)
            rx, rz = self.getReceiverCoords(isrc)
            Aop = AcousticWave2D(shape=shape, origin=origin, spacing=spacing, vp=vp0, nbl=nbl, space_order=space_order,
                                 src_x=sx, src_z=sz, rec_x=rx, rec_z=rz, t0=t0, tn=tn, src_type=src_type, f0=f0, dtype=dtype,
                                 op_name="fwd")
            Aops.append(Aop)
        return VStack(Aops)

    def getOperator(self, shape, origin, spacing, vp0, nbl, space_order, t0, src_type, f0, dtype, chunk=1, src0_idx=1, nshots=None):
        """
        The idea is that each operator is directly related to a shot present in the SEGY file, so we will not have
        problems with the dimensions of the output data. The basic idea would be a VStack containing
        nshots operators. However, the size of the data can exceed the computer's memory capacity. To work around
        this problem, it was defined that a list of VStacks would be created where each VStack would contain a specific
        number of operators (chunk).

        Thus, chunk defines how many shots/AcousticWave2D operators each VStack will encompass.
        """
        tn = self.getTn()

        if chunk == 1:
            return self._getOperatorUnique(shape, origin, spacing, vp0, nbl, space_order, t0, tn, src_type, f0, dtype, src0_idx, nshots)
        return self._getOperatorChunk(shape, origin, spacing, vp0, nbl, space_order, t0, tn, src_type, f0, dtype, chunk, src0_idx, nshots)
