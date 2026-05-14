import configargparse 
import json

def get_config():
    parser = configargparse.ArgumentParser()
    #-------------------------------- data config
    parser.add_argument("--voxel_grid_res", type=int, default=256, help='')
    parser.add_argument("--test_path", type=str, help='path', default='')
    parser.add_argument("--train_path", type=str, help='path', default='')
    parser.add_argument('--train_frames',type=int,default=1000)
    parser.add_argument('--test_frames',type=int,default=100)
    #-------------------------------- network config
    parser.add_argument("--conv_type", type=str, default='quant', help='')
    #---------------------------------- train config
    parser.add_argument("--device", type=str, default='cuda', help='')
    parser.add_argument("--log_path", type=str, default='output/debug', help='')
    parser.add_argument("--batch_size", type=int, default=1, help='')
    parser.add_argument("--n_epoch", type=int, default=50, help='')
    parser.add_argument("--val_frequence", type=int, default=10, help='')
    parser.add_argument("--lr", type=float, default=1e-4, help='')
    parser.add_argument("--init_ckpt", type=str, default='', help='')
    parser.add_argument('--mse_weight', type=float, default=1)
    parser.add_argument('--bce_weight', type=float, default=0.01)
    parser.add_argument('--entropy_weight', type=float, default=0.01)
    parser.add_argument('--only_test', type=int, default=0)

    return parser


def save_config(file_name, args):
    with open(file_name, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
