from RL_samp.header import *
from RL_samp.utils import fft_observe, mask_naiveRand, mask_prob, NRMSE, lpnorm, ssim_uniform

# import unet_model
from unet.unet_model import UNet
from unet.unet_model_fbr import Unet
from unet.unet_model_banding_removal_fbr import UnetModel

class dataloader():
    
    def __init__(self,data_dir='/home/ec2-user/SageMaker/data/OCMR_fully_sampled_images/',
                 unet_inchans=2,budget=32,base=8,
                 datatype=torch.float, eps=3, train_percentage=0.7, ncfiles=None, train_mode=True,
                 rand_mask_percentage=.5,device=torch.device('cpu')):
        self.data_dir = Path(data_dir)
        if ncfiles is None:
            ncfiles = list([])
            for file in os.listdir(data_dir):
                if file.endswith(".pt") and file.startswith('fs'):
                    ncfiles.append(file)
            random.shuffle(ncfiles)
            print('Number of useful files: ', len(ncfiles))
            train_file_num = int(np.ceil(train_percentage*len(ncfiles)))
            print('Number of Train  files: ', train_file_num)
            self.files = ncfiles[0:train_file_num]
        else:
            self.files = ncfiles
        
        self.filenum = len(self.files)
       
        self.base   = base
        self.budget = budget
        self.eps    = eps
        self.rand_mask_percentage = rand_mask_percentage
        self.unet_inchans = unet_inchans
        self.datatype = datatype
        self.train_mode = train_mode
        self.file_ind = 0
        self.device = device
    
    def select_file(self):
        if self.train_mode:
            ind = np.random.choice(len(self.files))
            self.curr_file = Path(self.files[ind])
#             print(self.curr_file)
        else:
            if self.file_ind < len(self.files):
                self.curr_file = Path(self.files[self.file_ind])
                self.file_ind += 1
            else:
                self.curr_file = None
    
    def reset(self):
        self.file_ind = 0
    
    def normalize(self,data,slices,reps):
        '''
        input data [:,:,0,t,0,slice,rep], all real images
        '''
        for sInd in range(slices):
            for repInd in range(reps):
                factor = torch.max(torch.abs(data[:,:,0,:,0,sInd,repInd]))
                data[:,:,0,:,0,sInd,repInd] = data[:,:,0,:,0,sInd,repInd]/factor
        return data
    
    def load(self):
        self.select_file() # randomly select a file
#         print(self.curr_file)
        if self.curr_file is not None:
            curr_data = torch.load(self.data_dir / self.curr_file)
            [H,W,_,self.t_ubd,_,self.slice_ubd,self.rep_ubd] = curr_data.shape

            curr_data = self.normalize(curr_data,self.slice_ubd,self.rep_ubd)
            curr_data = torch.moveaxis(curr_data,(2,3,4,5,6),(0,1,2,3,4))
            curr_data = torch.squeeze(curr_data)
            curr_data = torch.reshape(curr_data,(-1,H,W))
            curr_data = torch.unsqueeze(curr_data, 1)

            curr_fft = F.fftn(curr_data,dim=(2,3),norm='ortho')
            eps1 = random.randint(-self.eps, self.eps)
            eps2 = random.randint(-self.eps, self.eps)
            if np.random.rand() < self.rand_mask_percentage:
                mask = mask_naiveRand(W,fix=self.base+eps1,other=self.budget+eps2,roll=False) 
            else:
                mask = mask_prob(curr_fft,fix=self.base+eps1,other=self.budget+eps2,roll=False,fft_input=True)
            curr_fft[:,:,:,mask==0] = 0
            masked_data = F.ifftn(curr_fft,dim=(2,3),norm='ortho')
            if self.unet_inchans == 1:
                x_masked = torch.abs(masked_data)
            else:
                x_masked = torch.zeros(curr_fft.shape[0],2,H,W)
                x_masked[:,0,:,:] = torch.real(masked_data[:,0,:,:])
                x_masked[:,1,:,:] = torch.imag(masked_data[:,0,:,:])

            return x_masked.to(self.datatype).to(self.device), curr_data.to(self.datatype).to(self.device)
        
        else:
            self.reset()
            return None, None
    
class unet_trainer:
    def __init__(self,
                 net,
                 train_loader,
                 val_loader,
                 lr:float=1e-3,
                 lr_weight_decay:float=1e-8,
                 lr_s_stepsize:int=40,
                 lr_s_gamma:float=.1,
                 patience:int=5,
                 min_lr:float=5e-6,
                 reduce_factor:float=.8,
                 p=1,
                 weight_ssim:float=.7,
                 ngpu:int=1,
                 dir_checkpoint:str='/home/ec2-user/SageMaker/RLsamp/output/',
                 epochs:int=5,
                 infos:str=None,
                 max_batchsize:int=10,
                 ):
        self.ngpu   = ngpu
        self.device = torch.device('cuda:0') if ngpu > 0 else torch.device('cpu')
        self.net    = net.to(device)
        
        self.train_loader = train_loader
        self.val_loader   = val_loader
        
        self.lr = lr
        self.lr_weight_decay = lr_weight_decay
        self.lr_s_stepsize   = lr_s_stepsize
        self.lr_s_gamma      = lr_s_gamma
        self.patience        = patience
        self.min_lr          = min_lr
        self.reduce_factor   = reduce_factor
        self.p  = p
        self.weight_ssim     = weight_ssim
        self.dir_checkpoint  = dir_checkpoint

        self.infos  = infos
        self.epochs = epochs
        self.max_batchsize = max_batchsize
        self.hist   = []
        
        self.save_model = True
        self.train_df_loss = [] 
        self.train_ssim_loss = []
        self.train_loss_epoch = []
        self.val_df_loss = []
        self.val_ssim_loss = []
        self.val_loss_epoch = []
        self.valloss_old = np.inf
        
    def validate(self,epoch=-1):
        valloss = 0
        data_fidelity_loss = 0
        ssim_loss = 0
        
        self.net.eval()
        with torch.no_grad():
            x, xhat = self.val_loader.load()
            n_val = 0
            while x is not None:
                n_val += x.shape[0]
                
                ind_tmp = 0
                while ind_tmp < x.shape[0]:
                    if x.shape[0]<=self.max_batchsize:
                        x_tmp    = x
                        xhat_tmp = xhat
                        ind_tmp += x.shape[0]
                    else:
                        x_tmp    = x[ind_tmp:ind_tmp + self.max_batchsize]
                        xhat_tmp = xhat[ind_tmp:ind_tmp + self.max_batchsize]
                        ind_tmp += self.max_batchsize

                    pred = self.net(x_tmp).detach()

                    data_fidelity_loss += lpnorm(pred,xhat_tmp,p=self.p,mode='sum')
                    ssim_loss          += x_tmp.shape[0]-ssim_uniform(pred,xhat_tmp,reduction='sum') 
                x, xhat = self.val_loader.load()
            df_loss_epoch   = data_fidelity_loss.item()/n_val
            ssim_loss_epoch = ssim_loss.item()/n_val
            valloss_epoch   = df_loss_epoch  + self.weight_ssim * ssim_loss_epoch
            
            self.val_df_loss.append(df_loss_epoch)
            self.val_ssim_loss.append(ssim_loss_epoch)
            self.val_loss_epoch.append(valloss_epoch)
            if valloss_epoch < self.valloss_old:
                self.valloss_old = copy.deepcopy(valloss_epoch)
                self.save_model = True
            else:
                self.save_model = False
        print(f'\n\t[{epoch+1}/{self.epochs}]  loss/VAL: {valloss_epoch:.4f}, data fidelity loss: {df_loss_epoch:.4f} / 0, ssim loss: {ssim_loss_epoch:.4f} / 0')
        
        torch.cuda.empty_cache()
        return valloss_epoch
    
    def save(self,epoch=0,batchind=None):
        recName_base = self.dir_checkpoint + f'TrainRec_unet_fbr_{str(self.net.in_chans)}_chans_{str(self.net.chans)}'
        
        if self.infos is not None:
            recName_base = recName_base + self.infos
            
        recName = recName_base + '.npz'        
        np.savez(recName,trainloss_df=self.train_df_loss, 
                         trainloss_ssim=self.train_ssim_loss, 
                         trainloss_epoch=self.train_loss_epoch,
                         valloss_df=self.val_df_loss, 
                         valloss_ssim=self.val_ssim_loss, 
                         valloss_epoch=self.val_loss_epoch)
        print(f'\t History saved after epoch {epoch + 1}!')
        
        if (self.save_model) or (batchind is not None):
            
            modelName_base = self.dir_checkpoint + f'unet_fbr_{str(self.net.in_chans)}_chans_{str(self.net.chans)}'
            
            if self.infos is not None:
                modelName_base = modelName_base + self.infos           
            modelName = modelName_base + '.pt'
            
            torch.save({'model_state_dict': self.net.state_dict()}, modelName)  
            if batchind is None:
                print(f'\t Checkpoint saved after epoch {epoch + 1}!')
            else:
                print(f'\t Checkpoint saved at Python epoch {epoch}, batchnum {batchind}!')
                print('Model is saved after interrupt~')
            
        self.save_model = False
        torch.cuda.empty_cache()
    
    def run(self,
            save_cp=True):
        
        optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=self.lr_weight_decay)
        #         optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.lr_weight_decay)
        #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_s_stepsize, gamma=self.lr_s_gamma)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience, verbose=True, min_lr=self.min_lr,factor=self.reduce_factor)
        #         criterion = nn.MSELoss() 
        
#         breakpoint() 
        _ = self.validate(epoch=0)
        try:    
            for epoch in range(self.epochs):
                epoch_loss  = 0
                global_step = 1
                self.net.train()
                while global_step%self.train_loader.filenum > 0:
                    x, xhat = self.train_loader.load()
                    ind_tmp = 0
                    while ind_tmp < x.shape[0]:
                        if x.shape[0]<=self.max_batchsize:
                            x_tmp    = x
                            xhat_tmp = xhat
                            ind_tmp += x.shape[0]
                        else:
                            x_tmp    = x[ind_tmp:ind_tmp + self.max_batchsize]
                            xhat_tmp = xhat[ind_tmp:ind_tmp + self.max_batchsize]
                            ind_tmp += self.max_batchsize

                        pred = self.net(x_tmp)

                        data_fidelity_loss = lpnorm(pred,xhat_tmp,p=self.p,mode='mean')
                        ssim_loss = 1-ssim_uniform(pred,xhat_tmp,reduction = 'mean')
                        loss = data_fidelity_loss  + self.weight_ssim * ssim_loss

                        self.train_df_loss.append(data_fidelity_loss.item())
                        self.train_ssim_loss.append(ssim_loss.item())
                        epoch_loss += loss.item()

                        epoch_step = global_step%self.train_loader.filenum
                        print(f'[{epoch_step}/{self.train_loader.filenum}][{epoch+1}/{self.epochs}] loss/train: {loss.item():.4f}, data fidelity loss: {data_fidelity_loss.item():.4f}, ssim: {1-ssim_loss.item():.4f}')

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
                        optimizer.step()
                    
                        torch.cuda.empty_cache()
                        del x_tmp, xhat_tmp, pred
                    global_step += 1
                    del x, xhat, loss
                    torch.cuda.empty_cache()
                self.train_loss_epoch.append(epoch_loss/self.train_loader.filenum)
                
                valloss_epoch = self.validate(epoch=epoch)                
    #             scheduler.step()
                scheduler.step(valloss_epoch)

                if save_cp:  
                    self.save(epoch=epoch) 
        except KeyboardInterrupt:
            print('Keyboard Interrupted! Exit~')
            if save_cp:
                self.save(epoch=epoch)
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
            
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)   
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs', dest='epochs')
    
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-lrwd', '--lr-weight-decay', metavar='LRWD', type=float, nargs='?', default=0,
                        help='Learning rate weight decay', dest='lrwd')

    parser.add_argument('-utype', '--unet-type', type=int, default=2,
                        help='type of unet', dest='utype')
    
    parser.add_argument('-layer', '--unet-layer', metavar='LAYERS', type=int, nargs='?', default=6,
                        help='number of layers of unet', dest='unet_layers')
    
    parser.add_argument('-cn', '--channel-num', metavar='CN', type=int, nargs='?', default=64,
                        help='channel number of unet', dest='chans')
    parser.add_argument('-uc', '--uchan-in', metavar='UC', type=int, nargs='?', default=2,
                        help='number of input channel of unet', dest='in_chans')
    
    parser.add_argument('-b', '--max-batchsize', metavar='BATCH', type=int, nargs='?', default=10,
                        help='maximal batchsize', dest='max_batchsize')
    
    parser.add_argument('-sk','--skip',type=int,default=0,
                        help='residual network application', dest='skip')
    
    parser.add_argument('-bs','--base-size',metavar='BS',type=int,nargs='?',default=10,
                        help='number of observed low frequencies', dest='base_freq')
    
    parser.add_argument('-bg','--budget',metavar='BG',type=int,nargs='?',default=26,
                        help='number of high frequencies to sample', dest='budget')
    
    parser.add_argument('-s','--save-cp',metavar='SAVE',type=int,nargs='?',default=0,
                        help='save training result', dest='save_cp')
        
    parser.add_argument('-up', '--unet-path', type=str, default=None,
                        help='path file for a unet', dest='unetpath')
    parser.add_argument('-ngpu', '--num-gpu', type=int, default=1,
                        help='number of GPUs', dest='ngpu')
    
    parser.add_argument('-sd', '--seed', type=int, default=0,
                        help='random seed', dest='seed')
    parser.add_argument('-wssim', '--weight-ssim', metavar='WS', type=float, nargs='?', default=5,
                        help='weight of SSIM loss in training', dest='weight_ssim')
    
    parser.add_argument('-train-per', '--training-percentage', metavar='TP', type=float, nargs='?', default=.7,
                        help='percentage of files for training', dest='training_percentage')
    return parser.parse_args()
        
if __name__ == '__main__':  
    args = get_args()
    print(args)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    if args.skip == 0:
        skip = False
    else:
        skip = True
    
    if args.save_cp == 0:
        save_cp = False
    else:
        save_cp = True
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    
    if args.utype == 1:
        unet = UNet(in_chans=args.in_chans,n_classes=1,bilinear=(not skip),skip=skip).to(device)
    elif args.utype == 2: ## Unet from FBR
        unet = Unet(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0).to(device)
    elif args.utype == 3: ## Unet from FBR, res
        unet = UnetModel(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0,variant='res').to(device)
    elif args.utype == 4: ## Unet from FBR, dense
        unet = UnetModel(in_chans=args.in_chans,out_chans=1,chans=args.chans,num_pool_layers=args.unet_layers,drop_prob=0,variant='dense').to(device)
    
    if args.unetpath is not None:
        checkpoint = torch.load(args.unetpath)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print('Unet loaded successfully from: ' + args.unetpath )
    else:
        #         unet.apply(nn_weights_init)
        print('Unet is randomly initalized!')
    unet.train()        
    
    device = torch.device('cuda:0') if args.ngpu > 0 else torch.device('cpu')
    
    infos = f'base{args.base_freq}_budget{args.budget}'
    
    dir_checkpoint = '/home/ec2-user/SageMaker/RLsamp/output/'
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
    
    data_dir='/home/ec2-user/SageMaker/data/OCMR_fully_sampled_images/'
    ncfiles = list([])
    for file in os.listdir(data_dir):
        if file.endswith(".pt") and file.startswith('fs'):
            ncfiles.append(file)
    random.shuffle(ncfiles)
    print('Number of useful files: ', len(ncfiles))
    train_file_num = int(np.ceil(args.training_percentage*len(ncfiles)))
    print('Number of Train  files: ', train_file_num)
    print('Number of Val.   files: ', len(ncfiles)-train_file_num)
    train_files = ncfiles[0:train_file_num]
    val_files   = ncfiles[train_file_num:]
    
    train_dataloader = dataloader(base=args.base_freq, budget=args.budget, ncfiles=train_files, train_mode=True,device=device)
    val_dataloader   = dataloader(base=args.base_freq, budget=args.budget, ncfiles=val_files,  train_mode=False,device=device)
    trainer = unet_trainer(unet, train_dataloader, val_dataloader,
                           lr=args.lr,
                           lr_weight_decay=args.lrwd,
                           lr_s_stepsize=40,
                           lr_s_gamma=.8,
                           patience=10,
                           min_lr=1e-8,
                           reduce_factor=.8,
                           p='fro',
                           weight_ssim=args.weight_ssim,
                           ngpu=args.ngpu,
                           dir_checkpoint=dir_checkpoint,
                           epochs=args.epochs,
                           infos=infos,
                           max_batchsize=args.max_batchsize)
    
    trainer.run(save_cp=save_cp)