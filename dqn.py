from header import *
from utils  import *
from reconstructors import sigpy_solver

class DQN():
    '''
    policy: deep Q network
    assume processing one time series at a time
    '''
    def __init__(self,model,memory,
                      gamma:float=.5,
                      lr:float=.1,
                      eps:float=1e-3,
                      memory_len:int=3,
                      init_base:int=10,
                      L:float=5e-3,
                      max_iter:int=100,
                      solver:str='ADMM',
                      device=torch.device('cpu')):
        self.eps        = eps        
        self.model      = model
        self.memory     = memory
        self.init_base  = init_base
        self.gamma      = gamma
        self.lr         = lr
        self.optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        
        ## reconstructor parameters
        self.L = L
        self.max_iter = max_iter
        self.solver = 'ADMM'
        self.device = device
        
    def get_rand_action(self,mask):
        '''
        here the mask is 1D
        '''
        assert(len(mask.shape)==1)
        index_pool = (mask==0).nonzero(as_tuple=True)[0]
        loc = np.random.choice(index_pool,1)[0]
#         loc = index_pool[np.random.permutation(len(index_pool))[0]]
        return loc
    
    def get_action(self, data, eps_threshold, mask=None):
        '''
        here the mask is 1D
        '''
        assert(len(mask.shape)==1)
        dice = np.random.rand()
        if (dice < eps_threshold) and (mask is not None): # exploration with probability eps_threshold
            loc = self.get_rand_action(mask)
        else:
            res = self.model(data)
            loc = torch.argmax(res)
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
    
    def update_parameters(self):
        self.model.train()
        batch = self.memory.sample()
        if batch is None:
            return None
        ########################################
        #### Q-learning Bellman equation:
        #### min ||( Reward + gamma * max_a' Q(s',a') ) - Q_predicted||_2^2
        ########################################
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(1)
        
        ## Compute Q-values and get best action according to online network
        output_cur_step  = self.model(batch['observations'])
        all_q_values_cur = output_cur_step
        q_values = all_q_values_cur.gather(1, actions.unsqueeze(1))
        
        ## Compute target values using the best action found
        if self.gamma == 0.0:
            target_values = rewards
        else: ### HERE CAN BE ADJUSTED TO BE DOUBLE Q-LEARNING
            with torch.no_grad():
                ##
                # glue next_obs with curr_obs[1:]
                next_obs_last_slice = batch['next_observations']
                next_obs = torch.concat((batch['observations'][:,1:,:,:],next_obs_last_slice),dim=1)
                ##
                all_q_values_next = self.model(next_obs)
                target_values = torch.zeros(self.memory.batch_size, device=self.device)
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