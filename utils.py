from header import *

class ocmrLoader():
    def __init__(self,files,datapath='/mnt/shared_a/OCMR/OCMR_fully_sampled_images/',t_backtrack=3,batch_size=1,shuffle=True):
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
            self.fileIter = itertools.cycle(np.arange(len(self.ncfiles)))
        self.batch_size=batch_size
    
    def reset_iter(self):
        self.t     = copy.deepcopy(self.t_backtrack)
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
        if self.rep < self.rep_ubd:
            if self.slice < self.slice_ubd:
                curr_batchsize = min(self.batch_size,self.t_ubd-self.t-1)                
                data_source = torch.zeros(curr_batchsize,self.t_backtrack,self.curr_data.shape[0],self.curr_data.shape[1])
                data_target = torch.zeros(curr_batchsize,1,self.curr_data.shape[0],self.curr_data.shape[1])
                
                for ind in range(curr_batchsize):
                    data_source[ind,:,:,:] = self.curr_data[:,:,0,(self.t-self.t_backtrack):self.t,0,self.slice,self.rep].permute(2,0,1)
                    data_target[ind,:,:,:] = self.curr_data[:,:,0,self.t:self.t+1,0,self.slice,self.rep].permute(2,0,1)
                    self.t += 1 # time index increases by 1
                
                if self.t == self.t_ubd:
                    self.t = copy.deepcopy(self.t_backtrack)
                    self.slice += 1
                    if self.slice == self.slice_ubd:
                        self.rep  += 1
                        self.slice = 0                        
        if self.rep == self.rep_ubd:
            self.reset_iter()        
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
    
def fft_observe(imgs,mask,return_opt='img'):
    '''
    input imgs in image domain
    apply mask in the Fourier domain
    assume imgs in the shape [NCHW], mask in the shape [NW]
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
        return imgs_fft

def NRMSE(x,xstar):
        return torch.norm(x-xstar)/torch.norm(xstar)