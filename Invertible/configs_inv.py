import argparse
import os
def str2bool(x):
    return x.lower() in ('true')

#data_dir = "%s/lirh/ssd/dataset/shape_net_core_uniform_samples_2048"%ROOT

data_dir = "/home/lirh/ssd/dataset/Invertible"

feq = 20
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train',help="train/test")
parser.add_argument('--log_dir', default='log')
parser.add_argument('--dataset', default='model40')
parser.add_argument('--num_point', type=int, default=2048)
parser.add_argument('--num_sample_point', type=int, default=512)
parser.add_argument('--patch_num_point', type=int, default=2048)
parser.add_argument('--patch_num_ratio', type=int, default=3)

parser.add_argument('--sample_rate', type=int, default=4)



parser.add_argument('--num_sample_points', type=int, nargs='+', default=[64,128,256])
parser.add_argument('--data_dir', default=data_dir)
parser.add_argument('--restore', action='store_true')
parser.add_argument('--training_epoch', type=int, default=400)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--base_lr_d', type=float, default=0.0001)
parser.add_argument('--lr_decay', type=str2bool, default=True)
parser.add_argument('--lr_decay_rate', type=float, default=0.7)
parser.add_argument('--feq', type=float, default=feq)
parser.add_argument('--decay_steps', type=int, default=feq)
parser.add_argument('--start_decay_step', type=int, default=400)


parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch_per_save', type=int, default=feq)
parser.add_argument('--epoch_per_print', type=int, default=1)
parser.add_argument('--epoch_per_eval', type=int, default=5)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--visulize', type=str2bool, default=False)
parser.add_argument('--uniform', type=str2bool, default=False)
parser.add_argument('--mode', type=str, default='random', help='[uniform,random,partial]')
parser.add_argument('--use_repulse', type=str2bool, default=True)
parser.add_argument('--verbose', type=str2bool, default=True)
parser.add_argument('--supervised', type=str2bool, default=True)
parser.add_argument('--bneck_size', type=int, default=128)
parser.add_argument('--object_class', type=str, default='multi', help='Single class name (for example: chair) or multi [default: multi]')
parser.add_argument('--ae_dir', default='log/20191230-1755',help="ae_dir")
parser.add_argument('--ae_name', default='autoencoder')
parser.add_argument('--ae_restore_epoch', type=int, default=500)


FLAGS = parser.parse_args()

