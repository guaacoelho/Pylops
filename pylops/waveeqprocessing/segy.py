import segyio

import numpy as np
from scipy.spatial import distance


__all__ = ['count_segy_shots', 'get_velocity_model', 'ReadSEGY2D']


def count_segy_shots(segy_path, shotattr=segyio.TraceField.FieldRecord):
    with segyio.open(segy_path, "r", ignore_geometry=True) as segyfile:
        headers = segyfile.header
        shotpoints = [trace[shotattr] for trace in headers]

        shot_ids = set(shotpoints)
        return len(shot_ids), list(shot_ids)

def get_velocity_model(model_path):
    """
    Read velocity model from a SEGY file
    """
    f = segyio.open(model_path, iline=segyio.tracefield.TraceField.FieldRecord,
                    xline=segyio.tracefield.TraceField.CDP)

    xl, il, t = f.xlines, f.ilines, f.samples
    if len(il) != 1:
        dims = (len(xl), len(il), len(t))
    else:
        dims = (len(xl), len(t))

    vp = f.trace.raw[:].reshape(dims)
    return vp, dims

class ReadSEGY2D():

    def __init__(self, segy_path, mpi=None):

        self.segyfile = segy_path
        self.controller = mpi
        self.table, self.indexes = self.make_lookup_table(segy_path, mpi)
        #self.relative_distances = self._generate_relative_distances()
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

    def make_lookup_table(self, sgy_file, mpi_controller):
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
                index = header[segyio.TraceField.FieldRecord]
                
                if (mpi_controller and (index not in mpi_controller.shot_ids)):
                    pos_in_file += 1
                    continue                
                
                if int(header[segyio.TraceField.SourceGroupScalar]) < 0:
                    scalco = abs(1. / header[segyio.TraceField.SourceGroupScalar])
                else:
                    scalco = header[segyio.TraceField.SourceGroupScalar]
                # Esses comentários são temporários, scalel voltará a ser utilizado
                # if int(header[segyio.TraceField.ElevationScalar]) < 0:
                #     scalel = abs(1. / header[segyio.TraceField.ElevationScalar])
                # else:
                #     scalel = header[segyio.TraceField.ElevationScalar]
                # Check to see if we're in a new shot
                
                if index not in lookup_table.keys():
                    indexes.append(index)
                    lookup_table[index] = {}
                    lookup_table[index]['filename'] = sgy_file
                    lookup_table[index]['Trace_Position'] = pos_in_file
                    lookup_table[index]['Num_Traces'] = 1
                    lookup_table[index]['Source'] = (header[segyio.TraceField.SourceX] * scalco, header[segyio.TraceField.SourceY] * scalco)
                    lookup_table[index]['Receivers'] = []
                else:  # Not in a new shot, so increase the number of traces in the shot by 1
                    lookup_table[index]['Num_Traces'] += 1
                lookup_table[index]['Receivers'].append((header[segyio.TraceField.GroupX] * scalco, header[segyio.TraceField.GroupY] * scalco))
                pos_in_file += 1

        return lookup_table, indexes

    def _generate_relative_distances(self):
        # Retorna o menor valor de X e o valor de Y correspondente a esse receptor ou fonte
        minCoords = self.getMinXWithY()

        relative_dist = {minCoords[0]: 0.}
        for ii in self.indexes:
            # pega as coordenadas
            src_coords, rec_coords = self.getCoords(ii)
            x_all = np.concatenate((rec_coords[0], src_coords[0]))
            y_all = np.concatenate((rec_coords[1], src_coords[1]))

            for coordx, coordy in zip(x_all, y_all):
                if coordx in relative_dist:
                    continue
                coords = (coordx, coordy)
                relative_dist[coordx] = distance.euclidean(coords, minCoords)
        return relative_dist

    def getVelocityModel(self, path):
        """
        Read velocity model from a SEGY file
        """
        return get_velocity_model(path)

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

    def getRelativeCoords(self, index):
        src_coords, rec_coords = self.getCoords(index=index)

        new_rx = np.zeros((rec_coords[0].size,))
        new_rz = np.zeros((rec_coords[1].size,))
        for idx, coord in enumerate(rec_coords[0]):
            new_rx[idx] = self.relative_distances[coord]

        new_sx = np.array(self.relative_distances[src_coords[0][0]])
        new_sz = np.zeros((1,))
        return (new_sx, new_sz), (new_rx, new_rz)

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
            position = self.table[index]['Trace_Position']
            traces_in_shot = self.table[index]['Num_Traces']

            num_samples = len(f.samples)
            retrieved_shot = np.zeros((1, traces_in_shot, num_samples), dtype=np.float32)

            shot_traces = f.trace[position:position + traces_in_shot]

            for ii, trace in enumerate(shot_traces):
                retrieved_shot[:, ii] = trace
        return retrieved_shot

    def getMinCoords(self):
        """
        Get the origin of the survey. It is the minimum value of the source and receiver coordinates
        """
        minX = np.inf
        minY = np.inf
        for isrc in self.indexes:
            src_coords, rec_coords = self.getCoords(isrc)

            minX = min(minX, np.min(src_coords[0]), np.min(rec_coords[0]))
            minY = min(minY, np.min(src_coords[1]), np.min(rec_coords[1]))

        return minX, minY

    def getMinXWithY(self):
        """
        Retorna o menor valor de X entre fontes e receptores,
        junto com o valor de Y correspondente (mesmo índice).
        """
        minX = np.inf
        correspondingY = None

        for isrc in self.indexes:
            src_coords, rec_coords = self.getCoords(isrc)

            # Combina fontes e receptores
            x_all = np.concatenate((src_coords[0], rec_coords[0]))
            y_all = np.concatenate((src_coords[1], rec_coords[1]))

            # Encontra o menor X e o índice correspondente
            local_min_idx = np.argmin(x_all)
            local_minX = x_all[local_min_idx]

            # Se for menor que o atual mínimo, atualiza
            if local_minX < minX:
                minX = local_minX
                correspondingY = y_all[local_min_idx]

        return minX, correspondingY
