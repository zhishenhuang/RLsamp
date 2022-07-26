from header import *
from utils import *

class REINFORCE():
    def __init__(self,model,memory,
                    gamma:float=.5,
                    lr:float=.1,
                    traj:int=500,
                    horizon:int=500):
        self.model = model
        self.memory = memory
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.traj = traj
        self.horizon = horizon


    def get_rand_action(self,mask):
        '''
        here the mask is 1D
        '''
        assert(len(mask.shape)==1)
        index_pool = (mask==0).nonzero(as_tuple=True)[0]
        loc = np.random.choice(index_pool,1)[0]
#         loc = index_pool[np.random.permutation(len(index_pool))[0]]
        return loc
    
    def get_action(self, data, mask=None, eps_threshold:float=.1):
        '''
        here the mask is 1D
        '''
        dice = np.random.rand()
        if (dice < eps_threshold) and (mask is not None): # random policy for exploration with probability eps_threshold
            loc = self.get_rand_action(mask)
        else: # use model
            assert(len(mask.shape)==1)
            res = self.model(data,mask) # Jul 5, add mask here as second input
            loc = torch.argmax(res)
        return loc
    
    
    def step(self, action, target_gt, mask):
        '''
        action: [1] TODO: adding multiple lines at a time
        mask:[W]
        data:[N1HW]
        target:[N1HW]
        '''
        ### observe target_gt with old freq info
        target_obs_freq = fft_observe(target_gt,mask,return_opt='freq')
        img_recon = sigpy_solver(target_obs_freq, 
                                 L=self.L,max_iter=self.max_iter,solver=self.solver,
                                 heg=target_gt.shape[2],wid=target_gt.shape[3])
        old_nrmse = NRMSE(img_recon,target_gt)
        
        ### observe target_gt with new freq info
        mask[action] = 1 # incorporate action into mask
        target_obs_freq = fft_observe(target_gt,mask,return_opt='freq')
        next_obs = sigpy_solver(target_obs_freq, 
                                 L=self.L,max_iter=self.max_iter,solver=self.solver,
                                 heg=target_gt.shape[2],wid=target_gt.shape[3])
        new_nrmse = NRMSE(next_obs,target_gt)
        reward = old_nrmse - new_nrmse
        
        return next_obs, reward