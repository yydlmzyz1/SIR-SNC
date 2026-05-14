import os
import numpy as np
import torch
from tqdm import tqdm             
import time
import logging
from glob import glob
import random
import pandas as pd
import trimesh
from pathlib import Path
import sys
from config_load import get_config, save_config
import MinkowskiEngine as ME

# mesh utils
from sparse_rep import SparseTSDFRep, scale_model
from dataset import Dataset_SDF, make_data_loader

from utils_loss import get_bce_loss, get_mse_loss, get_entropy_loss, get_coords_bits
from utils_loss import test_distortion

class Trainer:

    def __init__(self, args):
        # log
        self.log_path = args.log_path
        os.makedirs(self.log_path,exist_ok=True)
        save_config(os.path.join(self.log_path,'config.txt'), args)
        self.logger = self.getlogger(logdir=self.log_path)
        self.record_set = {}

        # training settings
        self.n_epoch = args.n_epoch
        self.val_frequence = args.val_frequence
        self.only_test = args.only_test
        self.lr = args.lr

        # dataset
        if not self.only_test:
            file_list, _ = self.load_dataset(data_path=args.train_path, data_frames=args.train_frames)
            file_list = [item for sublist in file_list for item in sublist]
            self.dataset = Dataset_SDF(file_list=file_list, voxel_grid_res=args.voxel_grid_res)
            self.data_loader = make_data_loader(dataset=self.dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
        # test dataset
        file_list, name_list = self.load_dataset(data_path=args.test_path, data_frames=args.test_frames)
        self.test_data_loader_list = []
        self.test_dataname_list = name_list
        for sublist in file_list:
            test_dataset = Dataset_SDF(file_list=sublist, voxel_grid_res=args.voxel_grid_res)
            test_data_loader = make_data_loader(dataset=test_dataset, batch_size=1, shuffle=False, repeat=False)
            self.test_data_loader_list.append(test_data_loader)
        
        # network
        self.net = self.load_net(init_ckpt=args.init_ckpt)
        # optimizer
        self.optimizer = self.set_optimizer()
        # loss
        self.pruning = ME.MinkowskiPruning().to(args.device)
        self.bce_fn = torch.nn.BCEWithLogitsLoss().to(args.device)
        self.converter = SparseTSDFRep(args.voxel_grid_res)


    def getlogger(self, logdir=None):
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        if logdir:
            file_handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @torch.no_grad()
    def record(self, epoch, main_tag='train'):
        self.logger.info('='*10 + main_tag + ' Epoch ' + str(epoch))
        print('='*10 + main_tag + ' Epoch ' + str(epoch))
        for k, v in self.record_set.items():
            try:
                self.record_set[k]=np.mean(np.array(v), axis=0).squeeze()
            except:
                self.record_set[k] = v
        for k, v in self.record_set.items():
            self.logger.info(k+': '+str(np.round(v, 6)))
            print(k+': '+str(np.round(v, 6)))

        # return zero
        self.record_set = {}

        return
    
    def load_dataset(self, data_path, data_frames=5):
        # dataset
        p = Path(data_path)
        if p.suffix == '':
            p = p.with_suffix('.obj')
        if p.is_file():
            file_path = str(p)
            data_name = p.stem
            print('data_path', 0, file_path, data_name, 1)
            print('all_file_list', [data_name], [1])
            return [[file_path]], [data_name]

        all_file_list = []
        name_list = []
        data_path_list = data_path.split(' ')
        data_path_list = [f for f in data_path_list if f!='']
        
        for idx, data_path in enumerate(data_path_list):
            
            data_name = Path(data_path).resolve().parts[-1]
            name_list.append(data_name)

            file_list = glob(os.path.join(data_path, '**', f'*.obj'), recursive=True) + \
                        glob(os.path.join(data_path, '**', f'*.glb'), recursive=True)
            file_list = sorted(file_list)[:data_frames]
            print('data_path', idx, data_path, data_name, len(file_list)) 
            all_file_list.append(file_list)

        print('all_file_list', name_list, [len(f) for f in all_file_list])

        return all_file_list, name_list

    def set_optimizer(self):
        self.logger.info('='*5+' set_optimizer '+'='*5)
        print('='*5+' set_optimizer '+'='*5)
        params_lr_list = [] 
        for module_name in self.net._modules.keys():
            params_lr_list.append({"params":self.net._modules[module_name].parameters(), 'lr':self.lr})
            self.logger.info('optimize: '+module_name+'\tlr:'+str(self.lr))
            print('optimize: '+module_name+'\tlr:'+str(self.lr))
        optimizer = torch.optim.Adam(params_lr_list)
        
        return optimizer

    def load_net(self, init_ckpt=''):
        from network import JointAutoEncoder
        net = JointAutoEncoder(args).to(args.device)
        self.logger.info(net)
        # print(net)
        num_parmas = sum(p.numel() for p in net.parameters())
        bits_network = num_parmas * 8
        print('DBG!!!net parmas & size', num_parmas, int(bits_network/8192), 'KB')
        self.logger.info('DBG!!!net parmas & size ' + str(num_parmas) + ' ' + str(int(bits_network/8192))+'KB')
        # initialization
        if init_ckpt!='':
            self.logger.info('Load checkpoint from ' + init_ckpt)
            print('Load checkpoint from ' + args.init_ckpt)
            net = self.load_ckpt(net, args.init_ckpt)
        else:
            self.logger.info('Random initialization.')
            print('Random initialization.')

        return net

    def load_ckpt(self, net, init_ckpt):
        net_dict = net.state_dict()
        ckpt = torch.load(init_ckpt)
        pretrained_dict = {k:v for k,v in ckpt.items() if k in net_dict}
        pretrained_dict_keys = [k.split('.')[0] for k in pretrained_dict.keys()]
        pretrained_dict_keys = np.unique(pretrained_dict_keys).tolist()
        self.logger.info('Load pretained modules:' + str(pretrained_dict_keys))
        print('Load pretained modules:' + str(pretrained_dict_keys))

        ramdom_dict_keys = [k for k in net_dict.keys() if k not in pretrained_dict]
        ramdom_dict_keys = [k.split('.')[0] for k in ramdom_dict_keys]
        ramdom_dict_keys = np.unique(ramdom_dict_keys).tolist()
        self.logger.info('Random initialize modules:' + str(ramdom_dict_keys))
        print('Random initialize modules:' + str(ramdom_dict_keys))

        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
  
        return net
    
    def get_input(self, data_dict):
        coords_batch = data_dict['coords']
        feats_batch = data_dict['feats']

        coords, feats = ME.utils.sparse_collate(coords_batch, feats_batch)
        coords = coords.to(args.device)
        feats = feats.to(args.device)
        ground_truth = ME.SparseTensor(features=feats.float(), 
                                        coordinates=coords, 
                                        tensor_stride=1, 
                                        device=args.device)
        
        return ground_truth

    def forward(self, data_dict, training=True, DBG=False):
        # make input
        ground_truth = self.get_input(data_dict)

        # forward
        out_set = self.net(ground_truth=ground_truth, 
                           training=training)
        if out_set==None:
            return None, None
        
        # loss
        loss, record_set = self.get_loss(out_set=out_set, training=training)

        for k, v in record_set.items():
            if k not in self.record_set: self.record_set[k]=[]
            self.record_set[k].append(v)
        
        return out_set, loss

    def get_loss(self, out_set, training):
        loss = 0
        record_set = {}

        self.pooling_fn = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)

        # tsdf loss
        if args.mse_weight>0 and 'out_tsdf' in out_set.keys()and 'gt_tsdf' in out_set.keys():
            mse_loss, cur_record_set = get_mse_loss(gt=out_set['gt_tsdf'],
                                                    out=out_set['out_tsdf'],
                                                    pruning=self.pruning)
            loss += args.mse_weight * mse_loss
            record_set.update(cur_record_set)

        # occupancy loss
        if args.bce_weight>0 and 'out_cls_list' in out_set.keys() and 'gt_geo_list' in out_set.keys():
            bce_loss, cur_record_set = get_bce_loss(gt_list=out_set['gt_geo_list'],
                                                    cls_list=out_set['out_cls_list'],)
            
            loss += args.bce_weight * bce_loss
            record_set.update(cur_record_set)

        # entropy loss
        if args.entropy_weight>0 and 'likelihood' in out_set.keys():
            batch_size = out_set['gt_tsdf'].C[:, 0].max().item() + 1
            entropy_loss, cur_record_set = get_entropy_loss(likelihood=out_set['likelihood'], batch_size=batch_size)
            loss += args.entropy_weight * entropy_loss / 8192
            record_set.update(cur_record_set)

        if not training:
            save_coords = out_set['embed_features']
            cur_record_set = get_coords_bits(save_coords=save_coords, 
                                        save_path=args.log_path)
            record_set.update(cur_record_set)
            #        
            # record_set['feat_bits'] = len(out_set['bitstream'])*8/8192
            bin_dir = os.path.join(args.log_path, 'feats.bin')
            with open(bin_dir, 'wb') as f: f.write(out_set['bitstream'])
            record_set['feat_bits'] = os.path.getsize(bin_dir)*8/8192
            # sum
            record_set['sum_bits'] = record_set['feat_bits'] + record_set['coords_bits']

        return loss, record_set

    def train(self, epoch):
        for _, data_dict in enumerate(tqdm(self.data_loader)):
            _, loss = self.forward(data_dict=data_dict, training=True)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        torch.cuda.empty_cache()
        self.record(main_tag='Train', epoch=epoch)

        return

    @torch.no_grad()
    def test(self, epoch, val_data_loader, main_tag='Test'):
        out_path = args.log_path

        for index_t, data_dict in enumerate(tqdm(val_data_loader, desc= main_tag+' epoch %d'%epoch)):
            if index_t>args.test_frames: break
            # forward
            torch.cuda.empty_cache()
            out_set, _ = self.forward(data_dict=data_dict, training=False)

            # test_mesh
            test_results = self.test_mesh(out_set, data_dict, main_tag=main_tag)
            # update record
            for k, v in test_results.items():
                if k not in self.record_set: self.record_set[k]=[]
                self.record_set[k].append(v)
            
            # save to csv
            results_one = {'filedir':str(data_dict['files'][0])}
            for k in self.record_set.keys():
                v = self.record_set[k][-1]
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    for i in range(len(v)):
                        results_one[k+'-'+str(i)] = v[i]
                else:
                    results_one[k] = v
    
            self.logger.info('filedir:\t'+ str(data_dict['files'][0])+'\n'+str(results_one))
            print(results_one)
            results_one = pd.DataFrame([results_one])
            if index_t==0: 
                results = results_one.copy(deep=True)
            else: 
                try: results = results.append(results_one, ignore_index=True)
                except: results = results._append(results_one, ignore_index=True)
            results_filedir = os.path.join(out_path, main_tag+'.csv')
            results.to_csv(results_filedir, index=False)
            results_filedir = os.path.join(out_path, 'results.csv')
            results.to_csv(results_filedir, index=False)

        self.record(main_tag=main_tag, epoch=epoch)
        self.record_set = {}

        return
    
    @torch.no_grad()
    def test_mesh(self, out_set, data_dict, main_tag='Test'):
        out_path = args.log_path
        # os.makedirs(os.path.join(out_path, 'rec_data_'+main_tag), exist_ok=True)
        filename = os.path.split(data_dict['files'][0])[-1][:-4]

        # Rec Mesh 
        rec_meshfile = os.path.join(out_path, filename+'_rec.obj')
        out_tsdf = out_set['out_tsdf']
        vertices, faces = self.converter.sparseTSDF2mesh(out_tsdf.C[:,1:], out_tsdf.F)
        mesh_np = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), 
                                    faces=faces.detach().cpu().numpy(), process=False)
        mesh_np.export(rec_meshfile)

        # Ori Mesh
        ori_meshfile = os.path.join(out_path, filename+'_ori.obj')
        if not os.path.exists(ori_meshfile):
            raw_meshfile = data_dict['files'][0]
            ori_mesh = trimesh.load_mesh(raw_meshfile, file_type=raw_meshfile[-3:])
            verts, faces = scale_model(ori_mesh.vertices, ori_mesh.faces)
            mesh_np = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh_np.export(ori_meshfile)

        test_results = test_distortion(ori_meshfile, rec_meshfile)

        for k, v in test_results.items():
            if k not in self.record_set: self.record_set[k]=[]
            self.record_set[k].append(v)
        
        return test_results

    def run(self):
        for epoch in tqdm(range(1,self.n_epoch+1)):

            if (epoch%self.val_frequence==0 and epoch>0) or self.only_test:
                for test_dataname, test_data_loader in zip(self.test_dataname_list, self.test_data_loader_list):
                    self.test(epoch=epoch, val_data_loader=test_data_loader, main_tag='Test_'+test_dataname)
                # save network
                torch.save(self.net.state_dict(), os.path.join(self.log_path, 'model%04d.pt'%epoch)) 
                torch.save(self.net.state_dict(), os.path.join(self.log_path, 'model.pt'))            
            if self.only_test: break

            self.train(epoch)
            torch.save(self.net.state_dict(), os.path.join(self.log_path, 'model.pt'))

        return
    
if __name__=='__main__':
    args = get_config().parse_args()
    trainer = Trainer(args=args)
    trainer.run()
