import RL_samp
from RL_samp.header import *
from RL_samp.utils import *
from RL_samp.replay_buffer import *
from RL_samp.models import poly_net, val_net
from RL_samp.reconstructors import sigpy_solver
from RL_samp.policies import DQN
from RL_samp.trainers import DeepQL_trainer

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
    parser.add_argument('-base', '--base-sample', type=int, default=5,nargs='?',
                        help='base', dest='base')
    parser.add_argument('-bugdet', '--budget-sample', type=int, default=13,nargs='?',
                        help='budget', dest='budget')
    parser.add_argument('-gamma', '--discount-factor', type=float, default=.5,nargs='?',
                        help='discount factor', dest='gamma')
    
    parser.add_argument('-maxiter', '--max-iteration', type=int, default=50,nargs='?',
                        help='max iteration of sigpy solver', dest='maxiter')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    print(args)
    
    datapath = '/mnt/shared_a/OCMR/OCMR_fully_sampled_images/'
    ncfiles = list([])
    for file in os.listdir(datapath):
        if file.endswith(".pt"):
            ncfiles.append(file)
        
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
    save_freq   = 10
    batch_size  = args.batchsize
    ngpu        = 1
    lr          = args.lr
    eps         = 1e-3
    double_q    = False
    
    
    loader  = ocmrLoader(ncfiles,batch_size=1)
    memory  = ReplayMemory(capacity=memory_len,
                           curr_obs_shape=(t_backtrack,heg,wid),
                           mask_shape=(wid),
                           next_obs_shape=(1,heg,wid),
                           batch_size=batch_size,
                           burn_in=batch_size)
    model   = poly_net(samp_dim=wid)
    policy  = DQN(model,memory,max_iter=max_iter,ngpu=ngpu,gamma=discount,lr=lr,double_q_mode=double_q)
    trainer = DeepQL_trainer(loader,policy,episodes=episodes,
                             eps=eps,
                             base=base,budget=budget,
                             ngpu=ngpu)
    trainer.train()