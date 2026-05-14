import os
import numpy as np
import torch
import open3d as o3d
import argparse
import time
import pandas as pd
import time
from tqdm import tqdm
from glob import glob
import trimesh

from utils_tables import mc_table, num_triangles_table

cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [1,0,1], [0,0,1], [0,1,0], [1,1,0], [1,1,1], [0,1,1]], dtype=torch.int)


################################# utils #################################

def _sort_edges(edges):
    """sort last dimension of edges of shape (E, 2)"""
    with torch.no_grad():
        order = (edges[:, 0] > edges[:, 1]).long()
        order = order.unsqueeze(dim=1)

        a = torch.gather(input=edges, index=order, dim=1)
        b = torch.gather(input=edges, index=1 - order, dim=1)

    return torch.stack([a, b], -1)


def cal_sdf(vertices, faces, points):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points = points.astype(np.float32)
    vertices = vertices.astype(np.float32)

    vertices = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CPU:0"))
    faces = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32, device=o3d.core.Device("CPU:0"))
    mesh = o3d.t.geometry.TriangleMesh(vertices, faces)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CPU:0"))
    sdf_values = scene.compute_signed_distance(points)

    sdf_values = sdf_values.numpy()

    return sdf_values

base_cube_edges = torch.tensor([0,1,1,2,3,2,0,3,4,5,5,6,7,6,4,7,0,4,1,5,2,6,3,7], dtype=torch.long)
num_triangles_table = torch.tensor(num_triangles_table)
triangle_table_base = torch.tensor(mc_table, dtype=torch.long)

def marching_cubes_torch(vertices, cubes, sdf, res):
    device = vertices.device
    
    with torch.no_grad():
        occ_n = sdf > 0
        occ_fx8 = occ_n[cubes.reshape(-1)].reshape(-1, 8)
        occ_sum = torch.sum(occ_fx8, -1)
        valid_cubes = (occ_sum > 0) & (occ_sum < 8) 
        occ_sum = occ_sum[valid_cubes]
        #
        all_edges = cubes[valid_cubes][:, base_cube_edges.to(device)].reshape(-1, 2)
        all_edges = _sort_edges(all_edges)
        #
        if torch.__version__ == '1.10.0+cu111':
            unique_edges, idx_map = torch.unique(all_edges.cpu(), dim=0, return_inverse=True)
            unique_edges = unique_edges.to(all_edges.device)
            idx_map = idx_map.to(all_edges.device)
        else:
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]
        interp_v = unique_edges[mask_edges]
    
    if torch.__version__ == '1.10.0+cu111':
        edges_to_interp = vertices.cpu()[interp_v.cpu().reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp = edges_to_interp.to(device)
        edges_to_interp_sdf = sdf.cpu()[interp_v.cpu().reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf = edges_to_interp_sdf.to(device)
    else:
        edges_to_interp = vertices[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)
    
    # Check interpolation weights
    edges_to_interp_sdf[:, -1] *= -1
    denominator = edges_to_interp_sdf.sum(1, keepdim=True)
    safe_denominator = torch.where(denominator.abs() < 1e-8, torch.tensor(1e-8, device=device), denominator)
    edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / safe_denominator
    # edges interpation
    vertices = (edges_to_interp * edges_to_interp_sdf).sum(1)
    # Faces
    idx_map = idx_map.reshape(-1, 12)
    v_id = torch.pow(2, torch.arange(8, dtype=torch.long))
    cubeindex = (occ_fx8[valid_cubes] * v_id.to(device).unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table.to(device)[cubeindex]
    triangle_table = triangle_table_base.to(device)
    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1,
                    index=triangle_table[cubeindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1,
                    index=triangle_table[cubeindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 3], dim=1,
                    index=triangle_table[cubeindex[num_triangles == 3]][:, :9]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 4], dim=1,
                    index=triangle_table[cubeindex[num_triangles == 4]][:, :12]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 5], dim=1,
                    index=triangle_table[cubeindex[num_triangles == 5]][:, :15]).reshape(-1, 3),
    ), dim=0)
    scale_factor = 2.0 / (res - 1)
    vertices = vertices.float() * scale_factor - 1.0

    return vertices, faces

def scale_model(v,f, factor=1.99):
    v_min=np.min(v,axis=0)
    v_max=np.max(v,axis=0)

    center=(v_max+v_min)/2
    scale=np.max(v_max-v_min)

    v=(v-center)/scale*factor

    return v,f


################################### SparseTSDFRep ###################################
class SparseTSDFRep:
    def __init__(self, res, num_levels=5, threshold=1, device='cuda:0'):
        self.device = device

        self.res = res
        min_res = res / (2 **(num_levels - 1))
        self.res_list = [int(min_res*(2** i)) for i in range(num_levels)]
        # grids_set
        self.grids_set = {str(r): self.get_grids(threshold, r) for r in self.res_list}
        
    def get_grids(self, threshold, res):
        x = np.linspace(-threshold, threshold, res)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        X = X.reshape((np.prod(X.shape),))
        Y = Y.reshape((np.prod(Y.shape),))
        Z = Z.reshape((np.prod(Z.shape),))

        points_list = np.column_stack((X, Y, Z))
        del X, Y, Z, x
        grids=points_list.reshape(res,res,res,-1)
        
        return grids

    def mesh2sparseTSDF(self, vertices, faces, factor=2):
        # get mask of surface points
        for i, cur_res in enumerate(self.res_list):
            grids = self.grids_set[str(cur_res)]
            grids_flatten = grids.reshape(-1, 3)
            truncation = 2 / (cur_res - 1) * factor
            if i==0:
                sdf_flatten = cal_sdf(vertices, faces, grids_flatten)
                sdf = sdf_flatten.reshape(cur_res, cur_res, cur_res)
                mask = (np.abs(sdf) < truncation)
            else:
                mask = mask.repeat(2,axis=0).repeat(2,axis=1).repeat(2,axis=2)
                sdf = np.ones((cur_res,cur_res,cur_res),dtype=np.float32)*10000
                sel_grids = grids[mask]
                sdf_flatten = cal_sdf(vertices, faces, sel_grids)
                sdf[mask] = sdf_flatten
                mask = (np.abs(sdf) < truncation) * mask

        # calculate tsdf
        tsdf = np.clip(sdf / truncation, -1, 1)
        sparse_xyz = np.where(mask)
        sparse_tsdf = tsdf[sparse_xyz]
        sparse_xyz = np.stack(sparse_xyz, -1)
        
        surface_points = grids_flatten[mask.reshape(-1)]
        print(f"mesh2sparseTSDF surface_points: {len(surface_points)}")
        
        return sparse_xyz, sparse_tsdf
    

    def sparseTSDF2mesh(self, sparse_xyz, sparse_tsdf):

        if isinstance(sparse_xyz, np.ndarray):
            sparse_xyz = torch.from_numpy(sparse_xyz).to(self.device)
        if isinstance(sparse_tsdf, np.ndarray):
            sparse_tsdf = torch.from_numpy(sparse_tsdf).to(self.device)
                
        sparse_xyz = sparse_xyz.to(torch.int32)
        sparse_tsdf = sparse_tsdf.to(torch.float32)
        sparse_tsdf = sparse_tsdf.squeeze()

        device = sparse_xyz.device

        # cube_corners
        vertices = (cube_corners.unsqueeze(0).to(sparse_xyz) + sparse_xyz.unsqueeze(1))  # (V,8,3)
        mask = torch.zeros((vertices.size(0), vertices.size(1)), device=device)  # (V,8)
        mask[:, 0] = 1 
        vertices = vertices.reshape(-1, 3)
        mask = mask.reshape(-1)
        # 
        if torch.__version__ == '1.10.0+cu111':
            vertices_unique, inverse_indices = torch.unique(vertices.cpu(), dim=0, return_inverse=True)
            vertices_unique = vertices_unique.to(vertices.device)
            inverse_indices = inverse_indices.to(vertices.device)
        else:
            vertices_unique, inverse_indices = torch.unique(vertices, dim=0, return_inverse=True)
            
        cubes = inverse_indices.reshape(-1, 8)
        mask_unique = torch.zeros((vertices_unique.size(0)), dtype=torch.float32, device=device)
        mask_unique.scatter_add_(0, inverse_indices, mask)
        # 
        tsdf_unique = torch.zeros((vertices_unique.size(0)), dtype=torch.float32, device=device)
        tsdf_unique.scatter_(0, inverse_indices.reshape(-1, 8)[:, 0], sparse_tsdf)
        cube_mask = torch.sum(mask_unique[cubes.reshape(-1)].reshape(-1, 8), dim=-1)
        cubes = cubes[cube_mask == 8]
        # 
        vertices, faces = marching_cubes_torch(vertices_unique, cubes, tsdf_unique, res=self.res)
        
        return vertices, faces
    