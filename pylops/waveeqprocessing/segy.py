import segyio

import numpy as np
from pylops.basicoperators.vstack import VStack
# from pylops.waveeqprocessing.twoway import  AcousticWave2D


__all__ = ['ReadSEGY2D']


class ReadSEGY2D():

    def __init__(self, segy_path):

        self.segyfile = segy_path
        self.table, self.indexes = self.make_lookup_table(segy_path)
        self.isRecVariable = self._isRecVariable()
        self.nsrc = len(self.table)

    def set_shotID(self, id_src):
        self.id_src = id_src

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
        indexes = []
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
                    indexes.append(index)
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

        return lookup_table, indexes

    def getVelocityModel(self, path):
        """
        Read velocity model from a SEGY file
        """
        f = segyio.open(path, iline=segyio.tracefield.TraceField.FieldRecord,
                        xline=segyio.tracefield.TraceField.CDP)

        xl, il, t = f.xlines, f.ilines, f.samples
        if len(il) != 1:
            dims = (len(xl), len(il), len(t))
        else:
            dims = (len(xl), len(t))

        vp = f.trace.raw[:].reshape(dims)
        return vp, dims

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
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            num_samples = len(f.samples)
            samp_int = f.bin[segyio.BinField.Interval] / 1000

        return (num_samples - 1) * samp_int

    def getDt(self):
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            dt = f.bin[segyio.BinField.Interval]  # microseconds
        return dt / 1000  # return in miliseconds

    def getData(self, index: int):
        """
        Return the data from a specific index. It need to add a dimension to match returned data from _Wave
        Parameters
        ----------
        index : :obj:`int`
            Index of the shot that it will get the data
        """
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            retrieved_shot = []

            position = self.table[index]['Trace_Position']
            traces_in_shot = self.table[index]['Num_Traces']
            shot_traces = f.trace[position:position + traces_in_shot]

            for trace in shot_traces:
                retrieved_shot.append(trace)
        return np.expand_dims(np.array(retrieved_shot), axis=0)
