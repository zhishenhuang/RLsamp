import RL_samp
from RL_samp.header import *
from RL_samp.utils import *
from RL_samp.replay_buffer import *
from RL_samp.models import poly_net, val_net
from RL_samp.reconstructors import sigpy_solver
from RL_samp.policies import DQN
from RL_samp.trainers import DeepQL_trainer
from unet.unet_model import UNet
from unet.unet_model_fbr import Unet
from unet.unet_model_banding_removal_fbr import UnetModel


from importlib import reload

import argparse
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_args():
    parser = argparse.ArgumentParser(description='synthetic data setting',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,nargs='?',
                        help='learning rate', dest='lr')
    parser.add_argument('-mlen', '--memory-length', type=int, default=100,nargs='?',
                        help='length of memory in experience reply', dest='mlen')
    parser.add_argument('-e', '--epochs', type=int, default=1,nargs='?',
                        help='epoch', dest='epochs')
    parser.add_argument('-tb', '--t-backtrack', type=int, default=3,nargs='?',
                        help='t backtrack', dest='t_backtrack')
    parser.add_argument('-b', '--batchsize', type=int, default=3,nargs='?',
                        help='batchsize', dest='batchsize')
    parser.add_argument('-i', '--base-sample', type=int, default=5,nargs='?',
                        help='base', dest='base')
    parser.add_argument('-s', '--budget-sample', type=int, default=13,nargs='?',
                        help='budget', dest='budget')
    parser.add_argument('-gamma', '--discount-factor', type=float, default=.5,nargs='?',
                        help='discount factor', dest='gamma')
    
    parser.add_argument('-dq', '--double-q-mode', type=bool, default=True,nargs='?',
                        help='switch for double Q learning', dest='double_q')
    
    parser.add_argument('-maxiter', '--max-iteration', type=int, default=50,nargs='?',
                        help='max iteration of sigpy solver', dest='maxiter')
    parser.add_argument('-ngpu', '--num-gpu', type=int, default=1,nargs='?',
                        help='number of GPUs', dest='ngpu')
    parser.add_argument('-sfreq', '--save-frequency', type=int, default=10,nargs='?',
                        help='save frequency', dest='save_frequency')
    
    parser.add_argument('-utype', '--unet-type', type=int, default=2,
                        help='type of unet', dest='utype')
    parser.add_argument('-upath', '--unet-path', type=str, default=10,nargs='?',
                        help='unet_path', dest='unet_path')
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    print(args)
    
    datapath = '/home/ec2-user/SageMaker/data/OCMR_fully_sampled_images/'
    savepath = '/home/ec2-user/SageMaker/RLsamp/output/'
    ## import useful data files
    ncfiles = list([])
    for file in os.listdir(datapath):
        if file.endswith(".pt") and file.startswith('fs'):
            ncfiles.append(file)
    print('Number of useful files: ', len(ncfiles))
    
    if args.utype == 1:
        unet = UNet(in_chans=args.in_chans,n_classes=1,bilinear=(not skip),skip=skip).to(device)
    elif args.utype == 2: ## Unet from FBR
        unet = Unet(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0).to(device)
    elif args.utype == 3: ## Unet from FBR, res
        unet = UnetModel(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0,variant='res').to(device)
    elif args.utype == 4: ## Unet from FBR, dense
        unet = UnetModel(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0,variant='dense').to(device)
    
    if args.unet_path is not None:
        checkpoint = torch.load(args.unet_path)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print('Unet loaded successfully from: ' + args.unet_path )
    else:
        #         unet.apply(nn_weights_init)
        print('Unet is randomly initalized!')
    unet.train()   
        
    ## image parameters
    heg = 192
    wid = 144

    ## reconstructor parameters
    max_iter = args.maxiter
    L = 5e-3
    solver = 'ADMM'

    ## trainer parameters
    discount    = args.gamma
    memory_len  = args.mlen
    t_backtrack = args.t_backtrack
    base        = args.base
    budget      = args.budget
    episodes    = args.epochs
    save_freq   = args.save_frequency
    batch_size  = args.batchsize
    ngpu        = args.ngpu
    lr          = args.lr
    eps         = 1e-3
    double_q    = args.double_q
    
    
    loader  = ocmrLoader(ncfiles,batch_size=1,datapath=datapath)
    memory  = ReplayMemory(capacity=memory_len,
                           curr_obs_shape=(t_backtrack,heg,wid),
                           mask_shape=(wid),
                           next_obs_shape=(1,heg,wid),
                           batch_size=batch_size,
                           burn_in=batch_size)
    model   = poly_net(samp_dim=wid,in_chans=t_backtrack)
    policy  = DQN(model,memory,max_iter=max_iter,ngpu=ngpu,gamma=discount,lr=lr,double_q_mode=double_q,unet=unet)
    trainer = DeepQL_trainer(loader,policy,episodes=episodes,
                             eps=eps,
                             base=base,budget=budget,
                             ngpu=ngpu,
                             save_dir=savepath)
    trainer.train()