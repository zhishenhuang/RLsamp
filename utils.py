from header import *

# class ocmrLoader():
#     def __init__(self,files,datapath='/mnt/shared_a/OCMR/OCMR_fully_sampled_images/',
#                  t_backtrack=3,batch_size=1,shuffle=True,
#                  train_mode=True):
#         '''
#         [kx, ky, kz, phase/time, set, slice, rep]
#         frequency encoding, first phase encoding, second phase encoding, 
#         phase (time), set (velocity encoding), slice, repetition

#         better not to take average over the repetition dimension
#         '''
#         self.datapath    = datapath
#         self.t_backtrack = t_backtrack
#         self.files       = files       
#         if shuffle:
#             self.fileIter = itertools.cycle(np.random.permutation(len(self.files)))
#         else:
#             self.fileIter = itertools.cycle(np.arange(len(self.files)))
#         self.batch_size = batch_size
#         self.train_mode = train_mode
    
#     def reset_iter(self):
#         self.t     = copy.deepcopy(self.t_backtrack)
#         self.rep   = 0
#         self.slice = 0
#         self.curr_data = torch.load(self.datapath + self.curr_file)
#         self.t_ubd     = self.curr_data.shape[3]
#         self.slice_ubd = self.curr_data.shape[5]
#         self.rep_ubd   = self.curr_data.shape[6]
        
#     def reset(self):
#         self.curr_file = self.files[ self.fileIter.__next__() ]
#         print(f'current file: {self.curr_file}')
#         self.reset_iter()
    
#     def load(self):
#         if self.rep < self.rep_ubd:
#             if self.slice < self.slice_ubd:
#                 curr_batchsize = min(self.batch_size,self.t_ubd-self.t-1)                
#                 data_source = torch.zeros(curr_batchsize,self.t_backtrack,self.curr_data.shape[0],self.curr_data.shape[1])
#                 data_target = torch.zeros(curr_batchsize,1,self.curr_data.shape[0],self.curr_data.shape[1])
                
#                 for ind in range(curr_batchsize):
#                     data_source[ind,:,:,:] = self.curr_data[:,:,0,(self.t-self.t_backtrack):self.t,0,self.slice,self.rep].permute(2,0,1)
#                     data_target[ind,:,:,:] = self.curr_data[:,:,0,self.t:self.t+1,0,self.slice,self.rep].permute(2,0,1)
#                     self.t += 1 # time index increases by 1
                
#                 if self.t == self.t_ubd:
#                     self.t = copy.deepcopy(self.t_backtrack)
#                     self.slice += 1
#                     if self.slice == self.slice_ubd:
#                         self.rep  += 1
#                         self.slice = 0                        
#         if self.rep == self.rep_ubd:
#             if self.train_mode:
#                 self.reset_iter()
#             else:
#                 return None, None
#         return data_source, data_target
    
class ocmrLoader():
    def __init__(self,files,datapath='/mnt/shared_a/OCMR/OCMR_fully_sampled_images/',
                 t_backtrack=3,batch_size=1,shuffle=True,
                 train_mode=True):
        '''
        [kx, ky, kz, phase/time, set, slice, rep]
        frequency encoding, first phase encoding, second phase encoding, 
        phase (time), set (velocity encoding), slice, repetition

        better not to take average over the repetition dimension
        '''
        self.datapath    = datapath
        self.t_backtrack = t_backtrack
        self.files       = files       
        if shuffle:
            self.fileIter = itertools.cycle(np.random.permutation(len(self.files)))
        else:
            self.fileIter = itertools.cycle(np.arange(len(self.files)))
        self.batch_size = batch_size
        self.train_mode = train_mode
    
    def reset_iter(self):
        self.t     = 0
        self.rep   = 0
        self.slice = 0
        self.curr_data = torch.load(self.datapath + self.curr_file)
        self.t_ubd     = self.curr_data.shape[3]
        self.slice_ubd = self.curr_data.shape[5]
        self.rep_ubd   = self.curr_data.shape[6]
        
    def reset(self):
        self.curr_file = self.files[ self.fileIter.__next__() ]
        print(f'current file: {self.curr_file}')
        self.reset_iter()
        
    def load(self):
        '''
        load a batch of time series images
        '''
        if self.rep < self.rep_ubd:
            if self.slice < self.slice_ubd:               
                data_source = torch.zeros(self.batch_size,self.t_backtrack,self.curr_data.shape[0],self.curr_data.shape[1])
                data_target = torch.zeros(self.batch_size,1,self.curr_data.shape[0],self.curr_data.shape[1])
                
                for ind in range(self.batch_size):
                    endTime = self.t+self.t_backtrack
                    if endTime <= self.t_ubd-1:
                        data_source[ind,:,:,:] = self.curr_data[:,:,0,self.t:endTime   ,0,self.slice,self.rep].permute(2,0,1)
                        data_target[ind,:,:,:] = self.curr_data[:,:,0,endTime:endTime+1,0,self.slice,self.rep].permute(2,0,1)
                    else: # glue earlier frames onto the the current time series
                        source_part1 = self.curr_data[:,:,0,self.t:self.t_ubd,0,self.slice,self.rep].permute(2,0,1)
                        addOn        = self.t_backtrack - (self.t_ubd-self.t)
                        source_part2 = self.curr_data[:,:,0,0:addOn,0,self.slice,self.rep].permute(2,0,1)
                        data_source[ind,:,:,:] = torch.cat((source_part1,source_part2),dim=0)
                        data_target[ind,:,:,:] = self.curr_data[:,:,0,addOn:addOn+1,0,self.slice,self.rep].permute(2,0,1)
                    self.t += 1 # time index increases by 1    
                
                if self.t == self.t_ubd:
                    self.t = copy.deepcopy(self.t_backtrack)
                    self.slice += 1
                    if self.slice == self.slice_ubd:
                        self.rep  += 1
                        self.slice = 0
        if self.rep == self.rep_ubd:
            if self.train_mode:
                self.reset_iter()
            else:
                return None, None
        return data_source, data_target

def shiftsamp(sparsity,imgHeg):
    '''
    shiftsamp returns the sampled mask from the top and the bottom of an Image
    output: mask, maskInd, erasInd
    mask is a binary vector
    maskInd is the collection of sampled row markers
    erasInd is the collection of erased row markers
    '''
    assert(sparsity<=imgHeg)
    if sparsity <= 1:
        quota    = int(imgHeg*sparsity)
    else:
        quota    = int(sparsity)
    maskInd  = np.concatenate((np.arange(0,quota//2),np.arange(imgHeg-1,imgHeg-1-quota//2-quota%2,-1)))
    erasInd  = np.setdiff1d(np.arange(imgHeg),maskInd)
    mask     = torch.ones(imgHeg)
    mask[erasInd] = 0
    return mask,maskInd,erasInd

def mask_naiveRand(imgHeg,fix=10,other=30,roll=False):
    '''
    return a naive mask: return all known low-frequency
    while sample high frequency at random based on sparsity budget
    return UNROLLED mask!
    '''
    fix = int(fix)
    other = int(other)
    _, fixInds, _ = shiftsamp(fix,imgHeg)
    IndsLeft      = np.setdiff1d(np.arange(imgHeg),fixInds)
    RandInds      = IndsLeft[np.random.choice(len(IndsLeft),other,replace=False)]
    maskInd       = np.concatenate((fixInds,RandInds))
    erasInd       = np.setdiff1d(np.arange(imgHeg),maskInd)
    mask          = torch.ones(imgHeg)
    mask[erasInd] = 0
    if roll:
        mask = F.fftshift(mask)
    return mask
#     if not roll:
#         return mask,maskInd,erasInd
#     else:
#         mask = F.fftshift(mask)
#         return mask,None,None
    
def fft_observe(imgs,mask,return_opt='img',roll=False):
    '''
    input imgs in image domain
    apply mask in the Fourier domain
    assume imgs in the shape [NCHW], mask in the shape [NW]
    Aug 15: need to coordinate the convention between mask and fft info. 
            Since rolling fft info is only for the sigpy solver, so we only roll fft info but do not roll masks.
    '''
    imgs_fft = F.fftn(imgs,dim=(2,3),norm='ortho').to(torch.cfloat)
    if len(mask.shape) > 1:
        assert(imgs.shape[0]==mask.shape[0])
        for ind in range(len(imgs.shape[0])):
            imgs_fft[ind,:,:,mask[ind]==0] = 0
    else:
        imgs_fft[:,:,:,mask==0] = 0
    if return_opt == 'img':
        imgs_obs = torch.abs(F.ifftn(imgs_fft,dim=(2,3),norm='ortho'))
        return imgs_obs
    elif return_opt == 'freq':
        if roll:
            breakpoint()
            return F.fftshift(imgs_fft,dims=(2,3))
        else:
            return imgs_fft

def NRMSE(x,xstar):
        return torch.norm(x-xstar)/torch.norm(xstar)
    
# def encode_obs(curr_obs,mask) -> torch.Tensor:
#     batch_size, num_channels, img_height, img_width = curr_obs.shape
#     transformed_obs = torch.zeros(batch_size, num_channels, img_height+1, img_width).float()
#     transformed_obs[..., :img_height, :] = curr_obs
#     # The last row is the mask
#     transformed_obs[..., img_height, :]  = mask
#     return transformed_obs

# def decode_obs(obs):
#     full_height = obs.shape[2]
#     data = obs[..., :full_height, :]
#     # The last row is the mask
#     mask = obs[..., full_height, :]
#     return data, mask