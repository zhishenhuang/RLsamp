import RL_samp
from RL_samp.header import *
from RL_samp.utils  import *
from RL_samp.replay_buffer import *
from RL_samp.models import poly_net, val_net
from RL_samp.reconstructors import sigpy_solver
from RL_samp.trainers import AC1_ET_trainer

from unet.unet_model import UNet
from unet.unet_model_fbr import Unet
from unet.unet_model_banding_removal_fbr import UnetModel

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
    
    parser.add_argument('-lrv', '--step-size-val',  type=float, default=1e-5,nargs='?',
                        help='step size alpha for value network', dest='lrv')
    parser.add_argument('-lrp', '--step-size-poly', type=float, default=1e-5,nargs='?',
                        help='step size alpha for policy network', dest='lrp')
    
    parser.add_argument('-gamma', '--discount-factor', type=float, default=1-1e-5,nargs='?',
                        help='discount factor', dest='gamma')
    parser.add_argument('-slope', '--tanh-slope', type=float, default=.5,nargs='?',
                        help='slope for tanh activation', dest='slope')
    parser.add_argument('-rscale', '--reward-scale', type=float, default=1,nargs='?',
                        help='reward scale', dest='reward_scale')
    parser.add_argument('-vscale', '--valnet-scale', type=float, default=1,nargs='?',
                        help='valnet scale', dest='valnet_scale')
    
    parser.add_argument('-magweg', '--magnitude-weight', type=float, default=5,nargs='?',
                        help='magnitude weight', dest='mag_weight')
    
    parser.add_argument('-e', '--epochs', type=int, default=50,nargs='?',
                        help='epoch', dest='epochs')
    parser.add_argument('-tb', '--t-backtrack', type=int, default=3,nargs='?',
                        help='t backtrack', dest='t_backtrack')
    
    parser.add_argument('-base', '--base-sample', type=int, default=8,nargs='?',
                        help='base', dest='base')
    parser.add_argument('-bugdet', '--budget-sample', type=int, default=16,nargs='?',
                        help='budget', dest='budget')
    
    parser.add_argument('-maxiter', '--max-iteration', type=int, default=50,nargs='?',
                        help='max iteration of sigpy solver', dest='maxiter')
    
    parser.add_argument('-sfreq', '--save-frequency', type=int, default=10,nargs='?',
                        help='save frequency', dest='save_frequency')
    
    parser.add_argument('-utype', '--unet-type', type=int, default=2,
                        help='type of unet', dest='utype')
    parser.add_argument('-uc', '--uchan-in', metavar='UC', type=int, nargs='?', default=2,
                        help='number of input channel of unet', dest='in_chans')
    parser.add_argument('-layer', '--unet-layer', metavar='LAYERS', type=int, nargs='?', default=6,
                        help='number of layers of unet', dest='unet_layers')
    parser.add_argument('-cn', '--channel-num', metavar='CN', type=int, nargs='?', default=64,
                        help='channel number of unet', dest='chans')
    parser.add_argument('-upath', '--unet-path', type=str, default='/home/ec2-user/SageMaker/RLsamp/output/recon_models/unet_lowfreq_rand_0.0_fbr_2_chans_64base8_budget16.pt',nargs='?',
                        help='unet_path', dest='unet_path')
    parser.add_argument('-ulpath', '--unet-lowfreq-path', type=str, default='/home/ec2-user/SageMaker/RLsamp/output/recon_models/unet_lowfreq_rand_0.0_fbr_2_chans_64base8_budget16.pt',nargs='?',
                        help='unet_lowfreq_path', dest='unet_lowfreq_path')
    parser.add_argument('-urpath', '--unet-rand-path', type=str, default='/home/ec2-user/SageMaker/RLsamp/output/recon_models/unet_lowfreq_rand_1.0_fbr_2_chans_64base8_budget16.pt',nargs='?',
                        help='unet_rand_path', dest='unet_rand_path')
    
    parser.add_argument('-gep', '--guide-epoch', type=int, nargs='?', default=0,
                        help='guide Epochs', dest='guideEpoch')
    
    parser.add_argument('-ngpu', '--num-gpu', type=int, default=1,nargs='?',
                        help='number of GPUs', dest='ngpu')
    parser.add_argument('-gid', '--gpu-id', type=int, nargs='?', default=0,
                        help='GPU ID', dest='gpu_id')
    parser.add_argument('-istr', '--info-str', type=str, default=None,nargs='?',
                        help='info string to put in the saved data', dest='infostr')
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    print(args)
    
    device = torch.device('cpu') if args.ngpu==0 else torch.device(f"cuda:{args.gpu_id}")
    gpu_id1_alt = (args.gpu_id + 1)%4
    gpu_id2_alt = (args.gpu_id + 1)%4
    device_alt = torch.device('cpu') if args.ngpu==0 else [torch.device(f"cuda:{gpu_id1_alt}"),torch.device(f"cuda:{gpu_id2_alt}")]
    savepath = '/home/ec2-user/SageMaker/RLsamp/output/'
    datapath = '/home/ec2-user/SageMaker/data/OCMR_fully_sampled_images/'
#     ncfiles  = list([])
#     for file in os.listdir(datapath):
#         if file.endswith(".pt"):
#             ncfiles.append(file)
    ncfiles = np.load('/home/ec2-user/SageMaker/RLsamp/train_files.npz')['files']
    print('Number of Train files: ', len(ncfiles))
        
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
    
    ####################################################################################################
    if args.utype == 1:
        unet = UNet(in_chans=args.in_chans,n_classes=1,bilinear=(not skip),skip=skip).to(device_alt[0])
    elif args.utype == 2: ## Unet from FBR
        unet = Unet(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0).to(device_alt[0])
    elif args.utype == 3: ## Unet from FBR, res
        unet = UnetModel(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0,variant='res').to(device_alt[0])
    elif args.utype == 4: ## Unet from FBR, dense
        unet = UnetModel(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0,variant='dense').to(device_alt[0])
    
    if args.unet_path is not None:
        checkpoint = torch.load(args.unet_path)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print('Unet loaded successfully from: ' + args.unet_path )
    else:
        #         unet.apply(nn_weights_init)
        print('Unet is randomly initalized!')
    unet.eval()   
    
    ### Feb 20
    unet_lowfreq_path = args.unet_lowfreq_path
    unet_lowfreq = Unet(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0).to(device)
    checkpoint = torch.load(unet_lowfreq_path)
    unet_lowfreq.load_state_dict(checkpoint['model_state_dict'])
    
    
    unet_rand_path = args.unet_rand_path
    unet_rand = Unet(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0).to(device_alt[1])
    checkpoint = torch.load(unet_rand_path)
    unet_rand.load_state_dict(checkpoint['model_state_dict'])
    ####################################################################################################
    
    loader  = ocmrLoader(ncfiles,batch_size=1,datapath=datapath,t_backtrack=t_backtrack)
    p_net   = poly_net(samp_dim=wid,softmax=True,in_chans=t_backtrack)
    v_net   = val_net(slope=slope,scale=args.valnet_scale,in_chans=t_backtrack)
    trainer = AC1_ET_trainer(loader, polynet=p_net, valnet=v_net,
                             fulldim=wid, base=base, budget=budget,
                             lambda_poly=tdp, lambda_val=tdv,
                             alpha_poly=lrp,  alpha_val=lrv,
                             max_trajectories=episodes,
                             gamma=discount,
                             solver=solver, max_iter=max_iter, L=L,
                             device=device, device_alt=device_alt,
                             reward_scale=args.reward_scale,
                             save_dir=savepath,
                             unet=unet,
                             rand_eval_unet=unet_rand,
                             lowfreq_eval_unet=unet_lowfreq,
                             mag_weight=args.mag_weight,
                             infostr=args.infostr,
                             guide_epochs=args.guideEpoch)
    trainer.run()
    print(args)