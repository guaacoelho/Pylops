import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from pdb import set_trace

__all__ = ["MPIShotsController"]


class MPIShotsController:
    def __init__(self, shape, nsx, nsy, nbl, comm, shot_ids=None):
        
        self.shape = shape
        self.nsx = nsx
        self.nsy = nsy
        self.nbl = nbl
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.root = self.rank == 0
        self.global_shot_ids = shot_ids
        
        ls, le, nsl = self.divide_sdomain()
        self.local_start = ls
        self.local_end = le
        self.ns_local = nsl
        if shot_ids:
            self.shot_ids = shot_ids[ls:le]
        else: self.shot_ids = None
        
        

    def get_attr(self, name):
        attr = getattr(self, name, None)
        
        if not attr:
            raise ValueError(f"Attribute {attr} does not exist or not yet initialized")
        
        return attr
        
    def _crop_stencil(self, m, nbl):
        """Remove absorbing boundaries from model"""
        cropped_stencils = []
        for stencil in m:
            slices = tuple(slice(nbl, -nbl) for _ in range(stencil.ndim))
            cropped_stencils.append(stencil[slices])
        return cropped_stencils

    def divide_sdomain(self, M=None, n=None, idx=None):
        points = M or (self.nsx*self.nsy)
        size = n or self.size
        rank = idx or self.rank
        
        base = points // size
        remainder = points % size
        
        if rank < remainder:
            start = rank * (base + 1)
            end = start + (base + 1)
        else:
            start = remainder * (base + 1) + (rank - remainder) * base
            end = start + base
        
        return start, end, end-start
    
    
    def divide_shots(self, dpf=.05, mode="centered_uniform"):
        nsx = self.nsx
        nsy = self.nsy
        nshots = nsx*nsy
        nx, ny, nz = self.shape
        depth = int(dpf*nz)
        
        local_start, local_end, ns_local = self.divide_sdomain()
        
        local_sources = []
        
        if mode == "uniform":
            dx = (nx - 1) / (max(nsx - 1, 1)) if nsx > 1 else nx / 2
            dy = (ny - 1) / (max(nsy - 1, 1)) if nsy > 1 else ny / 2
            
            for sindex in range(local_start, local_end):
                iy = sindex // nsx
                ix = sindex % nsx
                
                x = ix * dx if nsx > 1 else dx
                y = iy * dy if nsy > 1 else dy
                z = depth
                
                local_sources.append([x, y, z])
        elif mode == "centered_uniform":
            d_sx = nx/(nsx+1) 
            d_sy = ny/(nsy+1) 
            for sindex in range(local_start, local_end):
                iy = sindex // nsx
                ix = sindex % nsx
                
                x = (ix + 1) * d_sx
                y = (iy + 1) * d_sy
                z = depth
                
                local_sources.append([x, y, z])
            
        self.local_start = local_start
        self.local_end = local_end
        self.ns_local = ns_local   
        return np.array(local_sources)
    
    def read_data(self, pwd, seismic_nt, mode="shot_number",
                  recs_names=["recX", "recY", "recZ"]):

        lstart = self.local_start
        lend = self.local_end
        nsx = self.nsx
        nx, ny, nz = self.shape
        
        dims_recs=[]
        for _ in recs_names:
            dims_recs.append([])
        
        for sindex in range(lstart, lend):
            sy = sindex // nsx
            sx = sindex % nsx
            for ri, rname in enumerate(recs_names):
                filename = f"{rname}_{sx}_{sy}.bin"
                aux = open(os.path.join(pwd, filename), "rb")
                data = np.fromfile(aux, dtype=np.float32, count=nx*ny*seismic_nt)
                aux.close()
                data = data.reshape([nx*ny, seismic_nt])
                dims_recs[ri].append(data)
        
        return dims_recs
    
    def write_data(self, recs_data, pwd, mode="shot_number",
                  recs_names=["recX", "recY", "recZ"]):

        lstart = self.local_start
        lend = self.local_end
        nsx = self.nsx
        
        shot = 0
        for sindex in range(lstart, lend):
            sy = sindex // nsx
            sx = sindex % nsx
            for ri, rname in enumerate(recs_names):
                filename = f"{rname}_{sx}_{sy}.bin"
                aux = open(os.path.join(pwd, filename), "wb")
                rec = recs_data[ri][shot]
                rec.tofile(aux)
                aux.close()
            shot+=1
    
    def build_result(self, local_image):
        shape = self.shape
        comm = self.comm
        
        if not isinstance(local_image, np.ndarray):
            local_image = np.array(local_image)
        
        result_sum = []
        for i in range(len(local_image)):
            result_dim = np.zeros(shape, dtype=np.float32)
            comm.Allreduce(local_image[i], result_dim, op=MPI.SUM)
            result_sum.append(result_dim)
        
        return np.array(result_sum)