import os
import numpy as np
import torch
import time
import MinkowskiEngine as ME
from utils_sparse import istopk, isin

from network_backbone import NetworkFactory, LinearLayers, ResNetBlock
from network_entropy import EntropyBottleneck


def downscale_sparse_tensor(ground_truth, down_scale, pooling_fn):
    gt = ME.SparseTensor(
        features=torch.ones([len(ground_truth), 1]).float(),
        coordinate_map_key=ground_truth.coordinate_map_key,
        coordinate_manager=ground_truth.coordinate_manager, device=ground_truth.device)

    gt_list = [gt]
    
    for _ in range(down_scale):
        gt = pooling_fn(gt)
        gt_list.append(gt)
    gt_list = gt_list[::-1]

    return gt_list


#########################################################################################
class OutOccupancyLayer(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer_cls = LinearLayers(channels, channels, 1)
        self.pruning_fn = ME.MinkowskiPruning()
    
    def forward(self, out, num_occupied, gt_geo, training=False):
        # get prob
        out_cls = self.layer_cls(out)
        prob = out_cls.F
        prob = torch.sigmoid(prob)

        # get topk
        mask = istopk(prob, k=int(num_occupied))

        if training:
            assert out.tensor_stride[0]==gt_geo.tensor_stride[0]
            mask_true = isin(out.C, gt_geo.C)# GT MASK
            mask = mask + mask_true

        out_pruned = self.pruning_fn(out, mask.to(out.device))

        return out_pruned, out_cls, gt_geo


#########################################################################################
class OutTSDFLayer(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer_tsdf = LinearLayers(channels, channels, 1)


    def forward(self, out, gt):
        out_set = {}
        if gt is not None: out_set['gt_tsdf'] = gt
        out_tsdf = self.layer_tsdf(out)
        out_set['out_tsdf'] = out_tsdf

        return out_set


#########################################################################################
class Encoder(torch.nn.Module):
    def __init__(self, in_channels=1, embed_channels=16, channels_list='16_16', block_layers=3, down_scale=2):
        super().__init__()
        if isinstance(channels_list, str):
            channels_list = [int(c) for c in channels_list.split('_')]
            channels_list = channels_list + [channels_list[-1]] * (down_scale+1-len(channels_list))
            channels_list = channels_list[:down_scale+1]
        self.down_scale = down_scale
 
        self.layer_in = LinearLayers(in_channels, channels_list[0], channels_list[0])
        self.layer_out = LinearLayers(channels_list[-1], channels_list[-1], embed_channels)

        self.block_list = torch.nn.ModuleList()
        for s in range(down_scale+1):
            self.block_list.append(ResNetBlock(
                in_channels=channels_list[s], channels=channels_list[s], 
                out_channels=channels_list[s], block_layers=block_layers))
    
        self.down_list = torch.nn.ModuleList()
        for s in range(down_scale):
            self.down_list.append(NetworkFactory.create_SparseConv3d(
                in_channels=channels_list[s], out_channels=channels_list[s+1], 
                kernel_size=2, stride=2, bias=True, dimension=3))

    def forward(self, ground_truth):
        embed_features = self.layer_in(ground_truth)
        embed_features = self.block_list[0](embed_features)
        for s in range(self.down_scale):
            embed_features = self.down_list[s](embed_features)
            embed_features = self.block_list[s+1](embed_features)
        embed_features = self.layer_out(embed_features)
 
        return embed_features  


#########################################################################################
class JointDecoder(torch.nn.Module):
    def __init__(self, embed_channels=16, channels_list='16_16', block_layers=3, down_scale=2):
        super().__init__()

        if isinstance(channels_list, str):
            channels_list = [int(c) for c in channels_list.split('_')]
            channels_list = channels_list + [channels_list[-1]] * (down_scale+1-len(channels_list))
            channels_list = channels_list[:down_scale+1]

        self.layer_in = LinearLayers(embed_channels, channels_list[0], channels_list[0])
        
        self.block_list = torch.nn.ModuleList()
        for s in range(down_scale+1):
            self.block_list.append(ResNetBlock(
                in_channels=channels_list[s], channels=channels_list[s], 
                out_channels=channels_list[s], block_layers=block_layers))

        self.up_list = torch.nn.ModuleList()
        for s in range(down_scale):

            self.up_list.append(NetworkFactory.create_SparseTransposeConv3d(
                in_channels=channels_list[s], out_channels=channels_list[s+1], 
                kernel_size=2, stride=2, bias=True, dimension=3,
                expand_coordinates=True))

        self.layer_cls_list = torch.nn.ModuleList() 
        for s in range(down_scale):
            self.layer_cls_list.append(OutOccupancyLayer(channels=channels_list[s+1]))

        self.layer_out = OutTSDFLayer(channels=channels_list[-1])

        self.down_scale = down_scale
        self.pooling_fn = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        self.pruning_fn = ME.MinkowskiPruning()

    def forward(self, embed_features, gt_geo_list, gt_tsdf, num_occupied_list, training=False):

        out_set = {'out_cls_list':[], 'gt_geo_list':[]}

        out = self.layer_in(embed_features)
        out = self.block_list[0](out)

        for s in range(self.down_scale):
            out = self.up_list[s](out)
            # block
            out = self.block_list[s+1](out)
            # cls
            layer_cls = self.layer_cls_list[s]
            num_occupied = num_occupied_list[s]
            if gt_geo_list is not None: gt_geo = gt_geo_list[s]
            else: gt_geo = None
            out, out_cls, gt_geo = layer_cls(out, num_occupied=num_occupied, gt_geo=gt_geo, training=training)
            out_set['out_cls_list'].append(out_cls)
            out_set['gt_geo_list'].append(gt_geo)

        tsdf_out_set = self.layer_out(out, gt=gt_tsdf)
        out_set.update(tsdf_out_set)

        return out_set


#########################################################################################
class JointAutoEncoder(torch.nn.Module):
    def __init__(self, args,
                 in_channels=1, embed_channels=16,
                 enc_channels_list='16_16_16', enc_block_layers=5, 
                 dec_channels_list='16_16_16', dec_block_layers=5, 
                 down_scale=2, voxel_grid_res=512):
        
        super().__init__()
        if args is not None: voxel_grid_res = args.voxel_grid_res
        if isinstance(enc_channels_list, str):
            enc_channels_list = [int(c) for c in enc_channels_list.split('_')]
            enc_channels_list = enc_channels_list + [enc_channels_list[-1]] * (down_scale+1-len(enc_channels_list))
            enc_channels_list = enc_channels_list[:down_scale+1]
        if isinstance(dec_channels_list, str):
            dec_channels_list = [int(c) for c in dec_channels_list.split('_')]
            dec_channels_list = dec_channels_list + [dec_channels_list[-1]] * (down_scale+1-len(dec_channels_list))
            dec_channels_list = dec_channels_list[:down_scale+1]

        self.encoder = Encoder(in_channels=in_channels, 
                               embed_channels=embed_channels,
                               channels_list=enc_channels_list, 
                               block_layers=enc_block_layers, 
                               down_scale=down_scale)
        
        self.decoder = JointDecoder(embed_channels=embed_channels,
                                    channels_list=dec_channels_list,
                                    block_layers=dec_block_layers, 
                                    down_scale=down_scale)
                
        self.down_scale = down_scale
        self.pooling_fn = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        # entropy model
        self.entropy_bottleneck = EntropyBottleneck(embed_channels)

        return

    def forward_entropy_model(self, embed_features, training):
        if training: quantize_mode = 'noise'
        else: quantize_mode = 'symbols'

        featQ, likelihood = self.entropy_bottleneck(embed_features.F, quantize_mode=quantize_mode)
        embed_features = ME.SparseTensor(features=featQ, 
                                coordinate_map_key=embed_features.coordinate_map_key, 
                                coordinate_manager=embed_features.coordinate_manager, 
                                device=embed_features.device)
        out_set = {'embed_features':embed_features, 'likelihood':likelihood}

        if not training:
            featQ = embed_features.F.round()
            strings, min_v, max_v = self.entropy_bottleneck.compress(featQ)
            # shape = featQ.shape
            # featQ_dec = self.entropy_bottleneck.decompress(strings, min_v, max_v, shape=shape, channels=shape[-1])
            # featQ_dec = featQ_dec.to(featQ.device)
            out_set['bitstream'] = strings

        return embed_features, out_set

    def forward(self, ground_truth, training=True):
        if not training:
            out_set_test = self.test(ground_truth)
            return out_set_test

        # gt_geo 
        gt_geo_list = downscale_sparse_tensor(ground_truth, 
                                              down_scale=self.down_scale-1,
                                              pooling_fn=self.pooling_fn)
        num_occupied_list = [gt_geo.C.shape[0] for gt_geo in gt_geo_list]

        # encode 
        embed_features = self.encoder(ground_truth=ground_truth)

        embed_features, enc_set = self.forward_entropy_model(
            embed_features=embed_features, training=training)

        # decode
        out_set = self.decoder(embed_features=embed_features,
                               gt_geo_list=gt_geo_list, 
                               gt_tsdf=ground_truth, 
                               num_occupied_list=num_occupied_list,
                               training=training)
        out_set.update(enc_set) 

        return out_set
    
    @torch.no_grad()
    def test(self, ground_truth):
        bitstream, coords, embed_features = self.encode(ground_truth)
        out_set = self.decode(bitstream, coords, device=ground_truth.device)
        out_set['out_tsdf'] = out_set['out_tsdf']
        out_set['embed_features'] = embed_features
        out_set['bitstream'] = bitstream
        
        return out_set
        
    @torch.no_grad()
    def encode(self, ground_truth):
        # gt_geo 
        gt_geo_list = downscale_sparse_tensor(ground_truth, 
                                              down_scale=self.down_scale-1,
                                              pooling_fn=self.pooling_fn)
        num_occupied_list = [gt_geo.C.shape[0] for gt_geo in gt_geo_list]

        embed_features = self.encoder(ground_truth=ground_truth)

        featQ = embed_features.F.round()
        strings, min_v, max_v = self.entropy_bottleneck.compress(featQ)
        shape = featQ.shape

        # pack bitstream
        bitstream = np.array(shape, dtype='int32').tobytes()
        bitstream += np.array(min_v, dtype='int16').tobytes()
        bitstream += np.array(max_v, dtype='int16').tobytes()
        bitstream += np.array(num_occupied_list[0], dtype='int32').tobytes()
        bitstream += np.array(num_occupied_list[1], dtype='int32').tobytes()
        bitstream += strings
        # coords
        coords = embed_features.C.cpu().numpy()[:,1:]//embed_features.tensor_stride[0]

        return bitstream, coords, embed_features
    
    @torch.no_grad()
    def decode(self, bitstream, coords, device):
        # unpack bitstream
        s = 0
        shape = np.frombuffer(bitstream[s:s+2*4], dtype='int32')
        s += 2*4
        min_v = np.frombuffer(bitstream[s:s+1*2], dtype='int16')
        s += 1*2
        max_v = np.frombuffer(bitstream[s:s+1*2], dtype='int16')
        s += 1*2
        num1 = np.frombuffer(bitstream[s:s+1*4], dtype='int32')
        s += 1*4
        num2 = np.frombuffer(bitstream[s:s+1*4], dtype='int32')
        s += 1*4
        num_occupied_list = [num1, num2]
        strings = bitstream[s:]

        # decompress
        featQ = self.entropy_bottleneck.decompress(strings, min_v, max_v, shape, channels=shape[-1])
        featQ = featQ.to(device)

        # embedded features
        tensor_stride = 4
        coords = torch.cat([torch.zeros([len(coords), 1]), 
                            torch.from_numpy(coords*tensor_stride)], dim=1)
        embed_features_dec = ME.SparseTensor(features=featQ, 
                                        coordinates=coords, 
                                        tensor_stride=tensor_stride, 
                                        device=device)

        # decode
        out_set = self.decoder(embed_features=embed_features_dec,
                               gt_geo_list=None,
                               gt_tsdf=None, 
                               num_occupied_list=num_occupied_list,
                               training=False)

        return out_set