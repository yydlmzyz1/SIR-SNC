import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import trimesh
from sparse_rep import SparseTSDFRep, scale_model
import time

class Dataset_SDF(Dataset):
    def __init__(self,  file_list=None, pin_memory=True, 
                voxel_grid_res=128):
        super().__init__()
                
        # mask_threshold = 1.1
        self.pin_memory=pin_memory
        self.voxel_grid_res=voxel_grid_res
        self.data_scale=1
        self.converter=SparseTSDFRep(res=self.voxel_grid_res)
 
        self.file_list = file_list
        self.data_list = []
        
        for i, f in enumerate(tqdm(self.file_list)):
            data = self.load_data(f)
            if isinstance(data, list):
                for d in data:
                    self.data_list.append(d)
            self.data_list.append(data)

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]

        return data

    def load_data(self, file_path):

        assert file_path.endswith('.obj') or file_path.endswith('.glb')
        folder_path, file_name = os.path.split(file_path)
        file_name = file_name.split('.')[0]
        file_pc = os.path.join(folder_path, file_name+'_res'+str(self.voxel_grid_res)+'_sdf.npz')
        
        if os.path.exists(file_pc):
            sdf_data = np.load(file_pc)                 
            sparse_xyz = sdf_data['xyz']
            sparse_tsdf = sdf_data['sdf']
        else:
            mesh = trimesh.load_mesh(file_path, file_type=file_path[-3:])
            verts = mesh.vertices
            faces = mesh.faces 
            verts, faces = scale_model(verts, faces)
            sparse_xyz, sparse_tsdf = self.converter.mesh2sparseTSDF(verts,faces)
            file_pc = os.path.join(folder_path, file_name+'_res'+str(self.voxel_grid_res)+'_sdf.npz')
            sparse_tsdf = sparse_tsdf.reshape(-1, 1)
            sparse_xyz = sparse_xyz.astype(np.int16)
            sparse_tsdf = sparse_tsdf.astype(np.float16)
            np.savez_compressed(file_pc, xyz=sparse_xyz, sdf=sparse_tsdf)

        return sparse_xyz, sparse_tsdf, file_path
         

###########################################################################################

class InfSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        else:
            perm = torch.range(0, perm-1).long()
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

def collate_fn(list_data):
    new_list_data = []
    for data in list_data:
        if data is not None: new_list_data.append(data)
    list_data = new_list_data
    if len(list_data)==0: raise ValueError('No data in the batch')
    coords, feats, files = list(zip(*list_data))

    return {'coords':coords, 'feats':feats, 'files':files}

def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=0, repeat=False, 
                    collate_fn=collate_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader

