from .header import *
    
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
        self.datapath     = datapath
        self.t_backtrack  = t_backtrack
        self.files        = files
        if shuffle:
            self.fileIter = itertools.cycle(np.random.permutation(len(self.files)))
        else:
            self.fileIter = itertools.cycle(np.arange(len(self.files)))
        self.batch_size   = batch_size
        self.train_mode   = train_mode
        
        self.reset_count  = 0
        self.file_count   = len(files)
            
    def normalize(self,data,slices,reps):
        '''
        input data [:,:,0,t,0,slice,rep], all real images
        '''
        for sInd in range(slices):
            for repInd in range(reps):
                factor = torch.max(torch.abs(data[:,:,0,:,0,sInd,repInd]))
                data[:,:,0,:,0,sInd,repInd] = data[:,:,0,:,0,sInd,repInd]/factor
        return data
    
    def reset_iter(self):
        self.t     = 0
        self.rep   = 0
        self.slice = 0
        self.curr_data = torch.load(self.datapath + self.curr_file)
        self.t_ubd     = self.curr_data.shape[3]
        self.slice_ubd = self.curr_data.shape[5]
        self.rep_ubd   = self.curr_data.shape[6]
        self.curr_data = self.normalize(self.curr_data,self.slice_ubd,self.rep_ubd)
        
    def reset(self):
        self.curr_file = self.files[ self.fileIter.__next__() ]
        print(f'current file: {self.curr_file}')
        self.reset_iter()
        print(f'Dimension of the current data file: t_ubd {self.t_ubd}, slice_ubd {self.slice_ubd}, rep_ubd {self.rep_ubd}')
        self.reset_count += 1
            
#     def test(self):
#         '''
#         load a batch of time series images
#         '''                        
#         for ind in range(self.batch_size):
#             print(f't {self.t}, rep {self.rep}, slice {self.slice}, batch ind {ind}')
#             endTime = self.t+self.t_backtrack
#             if endTime <= self.t_ubd-1:
#                 print(f'source inds: {np.arange(self.t,endTime)}, target inds: {endTime}')
#             else: # glue earlier frames onto the current last few frames in the time series
#                 addOn = self.t_backtrack - (self.t_ubd-self.t)
#                 print(f'source inds: {np.arange(self.t,endTime)}, target inds: {addOn}')
#             self.t += 1 # time index increases by 1    

#             if self.t == self.t_ubd:
#                 self.t = 0
#                 self.slice += 1
#                 print('\n  ~~ new slice ~~ \n')
#                 if self.slice == self.slice_ubd:
#                     print('\n  ~~ new rep ~~ \n')
#                     self.rep  += 1
#                     self.slice = 0
#             if self.rep == self.rep_ubd:
#                 print('\n  ~~ reset ~~ \n')
#                 if self.train_mode:
#                     self.reset_iter()
        
    def load(self):
        '''
        load a batch of time series images
            --> load() method can only be called after reset() is called at least once.
        '''
        if (self.rep == self.rep_ubd) and (not self.train_mode):
            return None, None
        data_source = torch.zeros(self.batch_size,self.t_backtrack,self.curr_data.shape[0],self.curr_data.shape[1])
        data_target = torch.zeros(self.batch_size,1,self.curr_data.shape[0],self.curr_data.shape[1])
        for ind in range(self.batch_size):               
            endTime = self.t+self.t_backtrack
            if endTime <= self.t_ubd-1:
                data_source[ind,:,:,:] = self.curr_data[:,:,0,self.t:endTime   ,0,self.slice,self.rep].permute(2,0,1)
                data_target[ind,:,:,:] = self.curr_data[:,:,0,endTime:endTime+1,0,self.slice,self.rep].permute(2,0,1)
            else: # glue earlier frames onto the current last few frames in the time series
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
            if (self.rep == self.rep_ubd): 
                self.reset()
                if not self.train_mode: # test_mode, still has useful data but less than batchsize
                    data_source = data_source[0:ind+1]
                    data_target = data_target[0:ind+1]
                    self.rep = self.rep_ubd
                    break
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
        quota = int(imgHeg*sparsity)
    else:
        quota = int(sparsity)
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

def mask_prob(imgs,fix=10,other=30,roll=True,fft_input=True):
    '''
    input imgs: [NCHW]
    '''
    fix   = int(fix)
    other = int(other)
    [N,C,_,sampdim] = imgs.shape
    if not fft_input:
        y = F.fftn(imgs,dim=(2,3),norm='ortho')
    else:
        y = imgs
    p = torch.sum(torch.abs(y),axis=(0,1,2))/(N*C)
    fixInds  = np.concatenate((np.arange(0,round(fix//2) ),np.arange(sampdim-1,sampdim-1-round(fix/2),-1)))
    p[fixInds] = 0
    p = p/p.sum() # normalize probability vector
    addInds  = np.random.choice(np.arange(sampdim),size=other,replace=False,p=p.numpy())
    maskInds = np.concatenate((fixInds,addInds))
    mask     = np.zeros(sampdim)
    mask[maskInds]= 1
    if roll:
        mask = np.roll(mask,shift=sampdim//2,axis=0)
    return mask
    
def fft_observe(imgs,mask,return_opt='img',roll=False, action=None, abs_opt=False):
    '''
    input imgs in image domain
    apply mask in the Fourier domain
    Input  format: imgs in the shape [NCHW], mask in the shape [NW]
    Output format: imgs in the shape [NCHW]
    Aug 15: need to coordinate the convention between mask and fft info. 
            Since rolling fft info is only for the sigpy solver, we only roll fft info but do not roll masks.
    Dec 29: if action is given, then return the magnitude of the newly sampled lines
    '''
    imgs_fft = F.fftn(imgs,dim=(2,3),norm='ortho').to(torch.cfloat)
    breakpoint()
    if action is not None:
        reference = torch.max(torch.sum(torch.abs(imgs_fft[:,:,:,0])**2, dim=2))
        [N,C,H,W] = imgs_fft.shape
        magnitude = torch.sum(torch.abs(imgs_fft[:,:,:,action])**2/reference) / (N*C)
        
    if len(mask.shape) > 1:
        assert(imgs.shape[0]==mask.shape[0])
        for ind in range(len(imgs.shape[0])):
            imgs_fft[ind,:,:,mask[ind]==0] = 0
    else:
        imgs_fft[:,:,:,mask==0] = 0
        
    if return_opt == 'img':
        imgs_obs = F.ifftn(imgs_fft,dim=(2,3),norm='ortho')
        if abs_opt:
            imgs_obs = torch.abs(imgs_obs)
        else:
            assert(C==1)
            img_output = torch.zeros(imgs_obs.shape[0],2,H,W)
            img_output[:,0,:,:] = torch.real(imgs_obs[:,0,:,:])
            img_output[:,1,:,:] = torch.imag(imgs_obs[:,0,:,:])
            img_obs = img_output
        if action is None:
            return imgs_obs
        else:
            return imgs_obs, magnitude
    elif return_opt == 'freq':
        if roll:
            if action is None:
                return F.fftshift(imgs_fft,dim=(2,3))
            else:
                return F.fftshift(imgs_fft,dim=(2,3)), magnitude
        else:
            if action is None:
                return imgs_fft
            else:
                return imgs_fft, magnitude    

def NRMSE(x,xstar):
        return torch.norm(x-xstar)/torch.norm(xstar)
    
def lpnorm(x,xstar,p='fro',mode='sum'):
    '''
    x and xstar are both assumed to be in the format NCHW
    '''
    assert(x.shape==xstar.shape)
    [N,C,H,W]   = x.shape
    numerator   = torch.norm(x-xstar,p=p,dim=(2,3))
    denominator = torch.norm(xstar  ,p=p,dim=(2,3))
    if   mode == 'sum':
        error = torch.sum( torch.div(numerator,denominator) ) / C
    elif mode == 'mean':
        error = torch.mean(torch.div(numerator,denominator) )
    elif mode == 'no_normalization':
        error = torch.mean(numerator)
    return error

#########################################################
# SSIM code from pputzky/irim_fastMRI/
#########################################################
def get_uniform_window(window_size, n_channels):
    window = torch.ones(n_channels, 1, window_size, window_size, requires_grad=False)
    window = window / (window_size ** 2)
    return window


def reflection_pad(x, window_size):
    pad_width = window_size // 2
    x = Func.pad(x, [pad_width, pad_width, pad_width, pad_width], mode='reflect')

    return x


def conv2d_with_reflection_pad(x, window):
    x = reflection_pad(x, window_size=window.size(-1))
    x = Func.conv2d(x, window, padding=0, groups=x.size(1))

    return x


def calc_ssim(x1, x2, window, C1=0.01, C2=0.03):
    """
    This function calculates the pixel-wise SSIM in a window-sized area, under the assumption
    that x1 and x2 have pixel values in range [0,1]. The default values for C1 and C2 are chosen
    in accordance with the scikit-image default values
    :param x1: 2d image
    :param x2: 2d image
    :param window: 2d convolution kernel
    :param C1: positive scalar, luminance fudge parameter
    :param C2: positive scalar, contrast fudge parameter
    :return: pixel-wise SSIM
    """
    x = torch.cat((x1, x2), 0)
    mu = conv2d_with_reflection_pad(x, window)
    mu_squared = mu ** 2
    mu_cross = mu[:x1.size(0)] * mu[x1.size(0):]

    var = conv2d_with_reflection_pad(x * x, window) - mu_squared
    var_cross = conv2d_with_reflection_pad(x1 * x2, window) - mu_cross

    luminance = (2 * mu_cross + C1 ** 2) / (mu_squared[:x1.size(0)] + mu_squared[x1.size(0):] + C1 ** 2)
    contrast = (2 * var_cross + C2 ** 2) / (var[:x1.size(0)] + var[x1.size(0):] + C2 ** 2)
    ssim_val = luminance * contrast
    ssim_val = ssim_val.mean(1, keepdim=True)

    return ssim_val

def ssim_uniform(input, target, window_size=11, reduction='mean'):
    """
    Calculates SSIM using a uniform window. This approximates the scikit-image implementation used
    in the fastMRI challenge. This function assumes that input and target are in range [0,1]
    input format: NCHW
    :param input: 2D image tensor
    :param target: 2D image tensor
    :param window_size: integer
    :param reduction: 'mean', 'sum', or 'none', see pytorch reductions
    :return: ssim value
    """
    window = get_uniform_window(window_size, input.size(1))
    window = window.to(input.device)
    ssim_val = calc_ssim(input, target, window)
    
    if reduction == 'mean':
        ssim_val = ssim_val.mean()
    elif not (reduction is None or reduction == 'none'):
        [N,C,H,W] = input.shape
        ssim_val = ssim_val.sum()/(C*H*W)

    return ssim_val
    
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