import os, sys, time
import numpy as np
import torch
import time
import pandas as pd
import torch.nn.functional as F
import math
import subprocess
import time
import os, sys
import trimesh
from pykdtree.kdtree import KDTree
import MinkowskiEngine as ME
from utils_sparse import isin, sort_sparse_tensor


bce_fn = torch.nn.BCEWithLogitsLoss()
ce_fn = torch.nn.CrossEntropyLoss()
softmax_fn = torch.nn.Softmax(dim=-1)


def get_bce(data, groud_truth):
    assert groud_truth.F.shape[-1]==1
    if len(data)==len(groud_truth):
        bce = bce_fn(data.F.squeeze(), groud_truth.F.squeeze())
    else:
        mask = isin(data.C, groud_truth.C)
        bce = bce_fn(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0] * groud_truth.shape[1]
    
    return sum_bce

def get_bce_loss(gt_list, cls_list):
    bce_loss, bce_loss_list = 0, [] 
    if len(cls_list)>0:
        for data_cls, gt in zip(cls_list, gt_list):
            if data_cls is None or gt is None: continue
            num_points = len(data_cls)
            curr_bce = get_bce(data_cls, gt) / num_points
            bce_loss += curr_bce
            bce_loss_list.append(curr_bce.item())

    with torch.no_grad():
        record_set = {}
        record_set['bce_loss'] = bce_loss.item() if isinstance(bce_loss, torch.Tensor) else bce_loss
        record_set['bce_loss_list'] =np.array(bce_loss_list)

    return bce_loss, record_set


def align_coordinates(gt, out, pruning):

    if out.C.shape[0]==gt.C.shape[0]:
        if (out.C==gt.C).all():
            return gt, out

    maskA = isin(out.C, gt.C).to(out.device)
    maskB = isin(gt.C, out.C).to(out.device)
    out_intersect = pruning(out, maskA)
    gt_intersect = pruning(gt, maskB)
    out_intersect = sort_sparse_tensor(out_intersect)
    gt_intersect = sort_sparse_tensor(gt_intersect)
    assert (gt_intersect.C==out_intersect.C).all()
    out_intersect = ME.SparseTensor(features=out_intersect.F, 
                                    coordinate_map_key=gt_intersect.coordinate_map_key,
                                    coordinate_manager=gt_intersect.coordinate_manager, 
                                    device=gt_intersect.device)

    return gt_intersect, out_intersect

def get_mse_loss(gt, out, pruning):
    assert gt.tensor_stride[0]==out.tensor_stride[0] 
    assert gt.shape[-1]==out.shape[-1] 
    gt_intersect, out_intersect = align_coordinates(gt, out, pruning)
    mse_loss = torch.mean(torch.abs(gt_intersect.F - out_intersect.F))
    
    with torch.no_grad():
        record_set = {}
        record_set['mse_loss'] = mse_loss.item()
        
    return mse_loss, record_set


def get_entropy_loss(likelihood, batch_size=1):
    entropy_loss = -torch.sum(torch.log2(likelihood))
    entropy_loss /= batch_size 

    with torch.no_grad():
        record_set = {}
        record_set['entropy_loss'] = entropy_loss.item() / 8192

    return entropy_loss, record_set


##########################################################################################

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item) 
        except ValueError: continue
        
    return number

rootdir_tmc13 = os.path.split(__file__)[0]

def gpcc_encode(filedir, bin_dir, posQuantscale=1, DBG=0):

    cmd = rootdir_tmc13+'/tmc3 --mode=0 ' \
        + ' --trisoupNodeSizeLog2=0' \
        + ' --neighbourAvailBoundaryLog2=8' \
        + ' --intra_pred_max_node_size_log2=6' \
        + ' --inferredDirectCodingMode=1' \
        + ' --maxNumQtBtBeforeOt=4' \
        + ' --minQtbtSizeLog2=0' \
        + ' --planarEnabled=1' \
        + ' --planarModeIdcmUse=0' \
        + ' --disableAttributeCoding=1' \
        + ' --positionQuantizationScale='+str(posQuantscale) \
        + ' --uncompressedDataPath='+filedir \
        + ' --compressedStreamPath='+bin_dir
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    subp.wait()
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    c=subp.stdout.readline()
    while c:
        if DBG: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline()

    return results

def gpcc_decode(bin_dir, dec_dir):
    cmd = rootdir_tmc13+'/tmc3 --mode=1 ' \
        + ' --compressedStreamPath='+bin_dir \
        + ' --reconstructedDataPath='+dec_dir \
        + ' --outputBinaryPly=0'
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    subp.wait()
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    c=subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value   
        c=subp.stdout.readline()

    return results


def write_ply_ascii_geo(filedir, coords, dtype='int32'):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype(dtype)
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 

    return


@torch.no_grad()
def get_coords_bits(save_coords, save_path=''):

    save_coords = save_coords.C[:,1:]/save_coords.tensor_stride[0]
    save_coords = save_coords.cpu().numpy()
    
    filedir = os.path.join(save_path, 'save_coords.ply')
    bin_dir = os.path.join(save_path, 'save_coords.bin')
    write_ply_ascii_geo(filedir=filedir, coords=save_coords)
    _ = gpcc_encode(filedir, bin_dir)
    bits_coords = os.path.getsize(bin_dir)*8
    
    record_set = {}
    record_set['coords_bits'] = bits_coords/8192

    return record_set


################################### test_distortion ###################################

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    points_src = points_src.astype(np.float32)
    points_tgt = points_tgt.astype(np.float32)
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src,)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def get_threshold_percentage(dist, thresholds):
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

def scale_mesh(in_file, scale=None, center=None):
    mesh = trimesh.load_mesh(in_file)
    vertices = mesh.vertices

    if scale is None:
        v_min=vertices.min(axis=0)
        v_max=vertices.max(axis=0)
        center=(v_max+v_min)/2
        scale=(v_max-v_min).max()
    # scaling
    vertices = (vertices- center)/scale*1.99
    mesh.vertices = vertices

    return mesh, scale, center

def eval_pointcloud(pointcloud,pointcloud_tgt,normals=None,normals_tgt=None,thresholds=np.linspace(1./1000,1,1000)):
    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)
    completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2
    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    chamferL1 = 0.5 * (completeness + accuracy)
    # F-Score
    F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]
    out_dict = {'chamfer-L1': chamferL1, 'f-score-5': F[4], 'normals': normals_correctness}


    return out_dict


def eval_mesh(mesh,gt_mesh, num_sample=100000):
    pointcloud, idx = mesh.sample(num_sample, return_index=True)

    pointcloud = pointcloud.astype(np.float32)
    normals = mesh.face_normals[idx]

    gt_pointcloud, gt_idx = gt_mesh.sample(num_sample, return_index=True)
    gt_pointcloud = gt_pointcloud.astype(np.float32)
    gt_normals = gt_mesh.face_normals[gt_idx]
    
    out_dict=eval_pointcloud(pointcloud,gt_pointcloud,normals,gt_normals)

    return out_dict


def test_distortion(meshA_file, meshB_file):
    
    meshA, scale, center = scale_mesh(meshA_file)
    meshB, _, _ = scale_mesh(meshB_file, scale=scale, center=center)

    eval_dict = eval_mesh(meshB, meshA)

    return eval_dict


def mean_dataframe(df_list):
    """calculate the average value of input df_list
    """
    df_mean = pd.DataFrame()
    for col in df_list[0].columns:
        try: 
            df_mean[col] = np.stack([df[col] for df in df_list]).mean(axis=0)
        except TypeError:
            continue

    return df_mean
