import os
import numpy as np
import torch
import MinkowskiEngine as ME


def array2vector(array, step=None):
    array, step = array.long().clone(), step.long().clone()
    if array.min()<0:
        min_value = array.min()
        array = array - min_value
        step = step - min_value
        
    assert array.min()>=0 and array.max()-array.min()<step
    array, step = array.long(), step.long()
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    data = data.clone()
    ground_truth = ground_truth.clone()
    device = data.device
    if len(ground_truth)==0:
        return torch.zeros([len(data)]).bool().to(device)
    # positive value
    min_value =  torch.min(data.min(), ground_truth.min())
    if min_value < 0:
        data[:,1:] -= min_value
        ground_truth[:,1:] -= min_value
    #
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = torch.isin(data.to(device), ground_truth.to(device))

    return mask

def istopk_local(data, k=1):
    mask = torch.zeros(len(data), dtype=torch.bool)
    _, indices = torch.topk(data.reshape(-1,8), k)
    indices += (torch.arange(0, len(indices))*8).reshape(-1,1).to(indices.device)
    indices = indices.reshape(-1)
    mask[indices] = True
    
    return mask.bool().to(data.device)

def istopk_global(data, k):
    mask = torch.zeros(len(data), dtype=torch.bool)
    k = min(k, len(data))
    _, indices = torch.topk(data.squeeze(), k)
    mask[indices] = True

    return mask.bool().to(data.device)

def istopk(prob, k):
    if prob.shape[0]%8==0:
        mask = istopk_local(prob, k=1)
        prob[torch.where(mask)[0]]=1
    else:
        print('ERROR!!! '*1000, 'prob.shape[0]%8!=0')
    mask_topk = istopk_global(prob, k=k)

    return mask_topk

def sort_sparse_tensor(sparse_tensor, target=None):
    if target is not None and (sparse_tensor.C==target.C).all():
        return ME.SparseTensor(features=sparse_tensor.F, 
                            coordinate_map_key=target.coordinate_map_key, 
                            coordinate_manager=target.coordinate_manager, 
                            device=target.device)

    # positive value
    coords = sparse_tensor.C.clone()
    min_value =  coords.min()
    if min_value < 0: coords[:,1:] -= min_value
    # sort
    indices = torch.argsort(array2vector(coords, coords.max()+1)).cpu()
    out_coords = sparse_tensor.C[indices]
    if torch.__version__=='1.10.0+cu111':
        out_feats = sparse_tensor.F.cpu()[indices]
    else:
        out_feats = sparse_tensor.F[indices]

    out = ME.SparseTensor(coordinates=out_coords, 
                        features=out_feats, 
                        tensor_stride=sparse_tensor.tensor_stride, 
                        device=sparse_tensor.device)

    if target is not None:
        # positive value
        target_coords = target.C.clone()
        min_value =  target_coords.min()
        if min_value < 0: target_coords[:,1:] -= min_value
        # sort
        target_indices = torch.argsort(array2vector(target_coords, target_coords.max()+1))
        inverse_indices = target_indices.sort()[1].cpu()
        assert (out_coords[inverse_indices]==target.C).all()
        out = ME.SparseTensor(features=out_feats[inverse_indices], 
                            coordinate_map_key=target.coordinate_map_key, 
                            coordinate_manager=target.coordinate_manager, 
                            device=target.device)

    return out
