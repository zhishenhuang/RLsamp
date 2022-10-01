import RL_samp
from RL_samp.header import *
from RL_samp.utils  import *
from RL_samp.replay_buffer import *
from RL_samp.models import poly_net, val_net
from RL_samp.reconstructors import sigpy_solver
from RL_samp.trainers import AC1_ET_trainer

import argparse
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_args():
    parser = argparse.ArgumentParser(description='policy gradient for dynamic MRI sampling',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-tdv', '--trace-decay-val',  type=float, default=.8,nargs='?',
                        help='trace decay rate lambda for value network', dest='tdv')
    parser.add_argument('-tdp', '--trace-decay-poly', type=float, default=.8,nargs='?',
                        help='trace decay rate lambda for policy network', dest='tdp')
    
    parser.add_argument('-lrv', '--step-size-val',  type=float, default=3e-1,nargs='?',
                        help='step size alpha for value network', dest='lrv')
    parser.add_argument('-lrp', '--step-size-poly', type=float, default=3e-1,nargs='?',
                        help='step size alpha for policy network', dest='lrp')
    
    parser.add_argument('-gamma', '--discount-factor', type=float, default=.5,nargs='?',
                        help='discount factor', dest='gamma')
    parser.add_argument('-slope', '--tanh-slope', type=float, default=.5,nargs='?',
                        help='slope for tanh activation', dest='slope')
    parser.add_argument('-rscale', '--reward-scale', type=float, default=9e2,nargs='?',
                        help='reward scale', dest='reward_scale')
    parser.add_argument('-vscale', '--valnet-scale', type=float, default=10,nargs='?',
                        help='valnet scale', dest='valnet_scale')
    
    parser.add_argument('-e', '--epochs', type=int, default=1,nargs='?',
                        help='epoch', dest='epochs')
    parser.add_argument('-tb', '--t-backtrack', type=int, default=3,nargs='?',
                        help='t backtrack', dest='t_backtrack')
    
    parser.add_argument('-base', '--base-sample', type=int, default=5,nargs='?',
                        help='base', dest='base')
    parser.add_argument('-bugdet', '--budget-sample', type=int, default=13,nargs='?',
                        help='budget', dest='budget')
    
    parser.add_argument('-maxiter', '--max-iteration', type=int, default=50,nargs='?',
                        help='max iteration of sigpy solver', dest='maxiter')
    
    parser.add_argument('-ngpu', '--num-gpu', type=int, default=1,nargs='?',
                        help='number of GPUs', dest='ngpu')
    parser.add_argument('-sfreq', '--save-frequency', type=int, default=10,nargs='?',
                        help='save frequency', dest='save_frequency')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    print(args)
    
    datapath = '/mnt/shared_a/OCMR/OCMR_fully_sampled_images/'
    ncfiles  = list([])
    for file in os.listdir(datapath):
        if file.endswith(".pt"):
            ncfiles.append(file)
        
    ## image parameters
    heg = 192
    wid = 144

    ## reconstructor parameters
    max_iter = args.maxiter
    L        = 5e-3
    solver   = 'ADMM' # ‘ConjugateGradient’, ‘GradientMethod’, ‘PrimalDualHybridGradient’

    ## trainer parameters
    discount    = args.gamma
    lrp         = args.lrp
    lrv         = args.lrv
    tdp         = args.tdp
    tdv         = args.tdv
    
    t_backtrack = args.t_backtrack
    base        = args.base
    budget      = args.budget
    episodes    = args.epochs
    save_freq   = args.save_frequency
    ngpu        = args.ngpu
    slope       = args.slope
    
    loader  = ocmrLoader(ncfiles,batch_size=1)
    p_net   = poly_net(samp_dim=wid,softmax=True)
    v_net   = val_net(slope=slope,scale=args.valnet_scale)
    trainer = AC1_ET_trainer(loader, polynet=p_net, valnet=v_net,
                             fulldim=wid, base=base, budget=budget,
                             lambda_poly=tdp, lambda_val=tdv,
                             alpha_poly=lrp,  alpha_val=lrv,
                             max_trajectories=episodes,
                             gamma=discount,
                             solver=solver, max_iter=max_iter, L=L,
                             ngpu=ngpu,reward_scale=args.reward_scale)
    trainer.run()