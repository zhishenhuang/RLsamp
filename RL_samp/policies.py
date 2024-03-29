from .header import *
from .utils  import *
from .reconstructors import sigpy_solver, unet_solver

class DQN():
    '''
    policy: deep Q network
    assume processing one time series at a time
    '''
    def __init__(self,model,memory,
                      gamma:float=.5,
                      lr:float=.1,
                      memory_len:int=3,
                      init_base:int=10,
                      L:float=5e-3,
                      max_iter:int=100,
                      solver:str='ADMM',
                      double_q_mode=False,
                      target_model=None,
                      device=torch.device('cpu'),
                      target_net_update_freq:int=20,
                      reward_scale:float=1.,
                      mag_weight:float=10.,
                      unet=None,
                      maxGuideEp=0):
        self.memory     = memory
        self.init_base  = init_base
        self.gamma      = gamma
        self.lr         = lr
        ## reconstructor parameters
        self.L = L
        self.max_iter = max_iter
        self.solver = 'ADMM'
        self.device = device
        self.reward_scale = reward_scale     
        self.model  = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.target_net_update_freq = target_net_update_freq
        self.double_q_mode = double_q_mode
        if double_q_mode:
            if target_model is None:
                target_model = copy.deepcopy(model)
            self.target_model = target_model.to(self.device)
        
        self.mag_weight = mag_weight
        self.unet = unet
        self.maxGuideEp = maxGuideEp
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
        with torch.no_grad():
            dice = np.random.rand()
            if (dice < eps_threshold) and (mask is not None): # random policy for exploration with probability eps_threshold
                loc = self.get_rand_action(mask)
            else: # use model
                assert(len(mask.shape)==1)
                res  = self.model(data.to(self.device),mask.to(self.device)) # Jul 5, add mask here as second input
                vals,locs = torch.max(res,dim=1)
                vals = vals / vals.sum()
                loc  = np.random.choice(locs.cpu().numpy(), 1, p=vals.cpu().numpy())[0]
            return loc
    
#     def compute_reward(self, kin, target,
#                        L=1e-4,max_iter=100,solver=None):
#         '''
#         option 1: sigpy
#         the mask inside obs should be UNROLLED!
#         input shape: kin [1,1,H,W], target [1,1,H,W]
#         '''
# #         kin = fft_observe(obs['img'],obs['mask'],return_opt='freq')
#         img_recon  = sigpy_solver(kin, L=L,max_iter=max_iter,heg=kin.shape[2],wid=kin.shape[3],solver=solver)
#         new_reward = NRMSE(img_recon,target) - self.curr_score
#         self.curr_score = NRMSE(img_recon,target)
#         return new_reward
    
    def step(self, action, target_gt, mask, epoch=0):
        '''
        action: [1] TODO: adding multiple lines at a time
        mask:[W]
        data:[N1HW]
        target:[N1HW]
        '''
        ### observe target_gt with old freq info
#         target_obs_freq, magnitude = fft_observe(target_gt,mask,return_opt='freq',roll=True, action=action) # changed, Dec 29
#         img_recon = sigpy_solver(target_obs_freq, 
#                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
#                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
        target_obs_freq, magnitude = fft_observe(target_gt, mask, return_opt='img', action=action, abs_opt=False)
        img_recon = unet_solver(target_obs_freq.to(self.device), self.unet) # target_obs_freq dim: [N, 2, H, W]
        old_nrmse = NRMSE(img_recon,target_gt.to(self.device))
        
        ### observe target_gt with new freq info
        mask[action] = 1 # incorporate action into mask
#         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True)
#         next_obs = sigpy_solver(target_obs_freq, 
#                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
#                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
        target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
        next_obs  = unet_solver(target_obs_freq.to(self.device), self.unet)
        new_nrmse = NRMSE(next_obs,target_gt.to(self.device))
        
        ## update reward: Dec 29, add the term giving weight to the magnitude of the added frequency line
        reward_extra = max(np.cos(epoch/self.maxGuideEp*np.pi/2),0) * self.mag_weight * magnitude if self.maxGuideEp > 0 else torch.tensor(0).to(self.device)
        reward = max(old_nrmse - new_nrmse, 0) + reward_extra
        
        return next_obs, reward*self.reward_scale, mask, [next_obs,target_gt.to(self.device)]
    
    def update_parameters(self):
        self.model.train()
        batch = self.memory.sample()
        if batch is None:
            return None
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
        ########################################
        #### Q-learning Bellman equation:
        #### min ||( Reward + gamma * max_a' Q(s',a') ) - Q_predicted||_2^2
        ########################################
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(1)
        ## Compute Q-values and get best action according to online network
        data = batch['observations']
        mask = batch['masks']
        output_cur_step  = self.model(data, mask=mask) # Jul 6: added mask
        all_q_values_cur = output_cur_step
        q_values = all_q_values_cur.gather(1, actions.unsqueeze(1))
        
        ## Compute target values using the best action found
        if self.gamma == 0.0:
            target_values = rewards
        else:
            with torch.no_grad():
                ### glue next_obs with curr_obs[1:]
                next_obs_last_slice = batch['next_observations']
                next_obs  = torch.concat((batch['observations'][:,1:,:,:],next_obs_last_slice),dim=1)
                next_mask = copy.deepcopy(mask)
                for ind in range(len(mask)):
                    next_mask[ind,actions[ind]] = 1
                ###
                if self.double_q_mode:
                    all_q_values_next = self.target_model(next_obs,mask=next_mask) # Jul 6: added mask
                else:
                    all_q_values_next = self.model(next_obs,mask=next_mask)
                target_values = torch.zeros(self.memory.batch_size)
                target_values = target_values.to(self.device)
                best_actions  = all_q_values_next.detach().max(1)[1]
                target_values = all_q_values_next.gather(1,best_actions.unsqueeze(1))
                target_values = self.gamma * target_values + rewards

        # loss = Func.mse_loss(q_values, target_values)
        loss = Func.smooth_l1_loss(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()

        # Compute total gradient norm (for logging purposes) and then clip gradients
        grad_norm: torch.Tensor = 0  # type: ignore
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
        self.optimizer.step()
        torch.cuda.empty_cache()
        
        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "q_values_mean": q_values.detach().mean().cpu().numpy(),
            "q_values_std": q_values.detach().std().cpu().numpy(),
        }
    