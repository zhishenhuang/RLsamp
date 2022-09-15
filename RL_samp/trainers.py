from .header import *
from .utils import fft_observe, mask_naiveRand

class DeepQL_trainer():
    def __init__(self,dataloader,policy,
                 episodes:int=10,
                 eps:float=1e-3,
                 fulldim:int=144,
                 base:int=10,
                 budget:int=50,
                 freq_dqn_checkpoint_save:int=10,
                 save_dir:str='/home/huangz78/rl_samp/',
                 ngpu=1):
        self.dataloader = dataloader
        self.dataloader.reset()
        
        self.policy   = policy
        self.episodes = episodes
        self.epi      = 0
        self.fulldim  = fulldim
        self.base     = base
        self.budget   = budget
        self.eps      = eps
        self.training_record = {'loss':[], 'grad_norm':[],'q_values_mean':[],'q_values_std':[],
                                'horizon_rewards':[]}
        self.steps    = 0
        self.save_dir = save_dir
        self.freq_dqn_checkpoint_save = freq_dqn_checkpoint_save
        self.ngpu     = ngpu
        
        if self.ngpu > 0:
            self.policy.model = self.policy.model.cuda()
        
    def train(self):  
        # run training
        while self.dataloader.reset_count//self.dataloader.file_count<self.episodes:
            print(f'episode [{self.dataloader.reset_count//self.dataloader.file_count}/{self.episodes}]')
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False)   
            # one mask at a time, start with a low frequency mask
            horizon_reward_total = 0
            while mask.sum() < self.budget + self.base:
                self.steps += 1
#                 print(f'step: {self.steps}, beginning, mask sum: {mask.sum().item()}')
                data_source, data_target = self.dataloader.load()
                mask_RL   = copy.deepcopy(mask)
                mask_rand = copy.deepcopy(mask)
#                 epsilon = _get_epsilon(steps_epsilon, self.options)
                curr_obs = fft_observe(data_source,mask_RL,return_opt='img')
                with torch.no_grad():
                    action = self.policy.get_action(curr_obs, mask=mask_RL, eps_threshold=self.eps)
#                     print(f'step {self.steps}, action {action}')
                    next_obs, reward, mask_RL = self.policy.step(action, data_target, mask_RL)
#                     print(f'step: {self.steps}, policy.step, mask_RL sum: {mask_RL.sum().item()}')

                    horizon_reward_total += reward
                    self.policy.memory.push(curr_obs, mask, action, next_obs, reward)
                    mask = copy.deepcopy(mask_RL)
#                     print(f'step: {self.steps}, assign, mask sum: {mask.sum().item()}')
                
                    ### compare with random policy
                    action_rand = self.policy.get_rand_action(mask=mask_rand)
                    _, reward_rand, mask_rand = self.policy.step(action_rand, data_target, mask_rand)
                ###
                
                update_results = self.policy.update_parameters()
                if update_results is not None:
                    for key in ['loss', 'grad_norm','q_values_mean','q_values_std']:
                        self.training_record[key].append(update_results[key])
                    curr_loss = update_results['loss']
                    print(f'step: {self.steps}, loss: {curr_loss:.4f}, RL reward: {reward.mean().item():.4f}, Rand reward: {reward_rand.mean().item():.4f} \n mask sum: {mask.sum().item()}')
                    torch.cuda.empty_cache()
                else:
                    print(f'step: {self.steps}, burn in, mask sum: {mask.sum().item()}')
                
                if self.steps % self.freq_dqn_checkpoint_save == 0:
                    self.save()
                if type(self.policy).__name__.lower() == 'ddqn':
                    if self.steps % self.policy.target_net_update_freq == 0:
                        self.target_net.load_state_dict(self.policy.model.state_dict())
            self.training_record['horizon_rewards'].append(horizon_reward_total)
#             self.dataloader.reset()
    
    def save(self):
        filename = f'{type(self.policy).__name__}_hist.pt'
        if type(self.policy).__name__.lower() == 'ddqn': 
            torch.save(
                    {
                        "dqn_weights": self.policy.model.state_dict(),
                        "target_weights": self.policy.target_net.state_dict(),
                        "training_record":self.training_record,
                    },
                    self.save_dir + filename,
                )
        elif type(self.policy).__name__.lower() == 'dqn':
            torch.save(
                    {
                        "dqn_weights": self.policy.model.state_dict(),
                        "training_record":self.training_record,
                    },
                    self.save_dir + filename,
                )
        print(f'At step {self.steps}, training history saved as {self.save_dir + filename}')
        
class RL_tester():
    def __init__(self,dataloader,policy,
                 fulldim:int=144,base:int=10,budget:int=50):
        self.dataloader = dataloader
        self.dataloader.reset()
        self.dataloader.train_mode = False
        self.policy   = policy
        self.epi      = 0
        self.episodes = self.dataloader.files
        self.fulldim  = fulldim # full size of the dimension to be sampled
        self.base     = base
        self.budget   = budget
        self.eps      = eps
        self.test_record = {'loss':[],'grad_norm':[],'q_values_mean':[],'q_values_std':[]}
        self.testRec  = np.zeros((self.dataloader.file_count))
        self.steps    = 0
    
    def run(self):  
        
        self.policy.model.eval()
        ### run testing
        with torch.no_grad():
            while self.dataloader.reset_count <= self.file_count:
                print(f'[file {self.dataloader.reset_count}/{self.file_count}]')
                mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False)
                epi_loss = 0
                slice_count = 0
                ### Build masks for the current time series
                while mask.sum() < self.budget + self.base:
                    self.steps += 1
                    data_source, data_target = self.dataloader.load()
    #                 epsilon = _get_epsilon(steps_epsilon, self.options)
                    curr_obs = fft_observe(data_source,mask)
                    action   = self.policy.get_action(curr_obs, mask=mask, eps_threshold=self.eps)
                    next_obs, _, mask = self.policy.step(action, data_target, mask)  # inplace mask change

                ### Do reconstruction 
                self.dataloader.reset_iter() # modifying while loop condition
                self.dataloader.batch_size = 1
                self.dataloader.train_mode = False

                while True:
                    data_source, data_target = self.dataloader.load()
                    if data_source is None:
                        break
                    curr_obs   = fft_observe(data_source,mask,return_opt='freq',roll=True)
                    img_recon  = sigpy_solver(curr_obs, 
                                             L=self.policy.L,
                                             max_iter=self.policy.max_iter,
                                             solver=self.policy.solver,
                                             heg=curr_obs.shape[2],wid=curr_obs.shape[3])
                    curr_nrmse = NRMSE(img_recon,data_source)
                    epi_loss    += curr_nrmse * curr_obs.shape[0]
                    slice_count += curr_obs.shape[0]

                self.testRec[self.epi] = epi_loss / slice_count
    #                 if self.steps % self.options.target_net_update_freq == 0:
    #                     self.logger.info("Updating target network.")
    #                     self.target_net.load_state_dict(self.policy.state_dict())
                self.dataloader.reset()
    
        return self.testRec

    
##############
# actor-critic 1 trainer, debug needed
##############
class AC1_trainer():
    def __init__(self, dataloader, polynet, valnet,
                  fulldim:int=144,base:int=10,budget:int=50,
                  gamma:float=.8,
                  horizon:int=None,
                  max_trajectories:int=100,
                  lr:float=.1,
                  init_base:int=10,
                  L:float=5e-3,
                  max_iter:int=100,
                  solver:str='ADMM',
                  device=torch.device('cpu'),
                  save_dir:str='/home/huangz78/rl_samp/'):
        self.dataloader = dataloader
        self.dataloader.reset()
        self.polynet    = polynet
        self.valnet     = valnet
        self.fulldim    = fulldim
        self.base       = int(base)
        self.budget     = int(budget)
        self.gamma      = gamma
        if horizon is None:
            self.horizon = self.budget
        else:
            self.horizon = int(horizon)
        self.lr         = lr
        self.optimizer_val  = optim.Adam(self.valnet.parameters(),  lr=lr)
        self.optimizer_poly = optim.Adam(self.polynet.parameters(), lr=lr)
        self.init_base  = int(init_base)
        self.max_trajectories = int(max_trajectories)
        ## reconstructor parameters
        self.L = L
        self.max_iter = int(max_iter)
        self.solver = 'ADMM'
        self.device = device
        
        self.reward_per_horizon = []
        self.save_dir = save_dir
    
    def get_action(self, curr_obs, mask=None):
        '''
        here the mask is 1D
        '''
        if mask is not None:
            assert(len(mask.shape)==1)
        res  = self.polynet(curr_obs,mask) # Jul 5, add mask here as second input
        loc  = torch.argmax(res)
        prob = res.gather(dim=1,index=loc.long().view(-1,1)).squeeze()
        return loc, prob
        
    def step(self, action, target_gt, mask):
        '''
        action: [1] TODO: adding multiple lines at a time
        mask:[W]
        data:[N1HW]
        target:[N1HW]
        '''
        ### observe target_gt with old freq info
        target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
        img_recon = sigpy_solver(target_obs_freq, 
                                 L=self.L,max_iter=self.max_iter,solver=self.solver,
                                 heg=target_gt.shape[2],wid=target_gt.shape[3])
        old_nrmse = NRMSE(img_recon,target_gt)
        
        ### observe target_gt with new freq info
        mask[action] = 1 # incorporate action into mask
        target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
        next_obs  = sigpy_solver(target_obs_freq, 
                                 L=self.L,max_iter=self.max_iter,solver=self.solver,
                                 heg=target_gt.shape[2],wid=target_gt.shape[3])
        new_nrmse = NRMSE(next_obs,target_gt)
        reward    = old_nrmse - new_nrmse
        
        return next_obs, reward
        
    def run(self):
        for trajectory in range(self.max_trajectories):
            
            print(f'trajectory [{trajectory+1}/{self.max_trajectories}]')
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False) # curr_state
            I = 1
            reward_horizon = 0
            for t in range(self.horizon):
                breakpoint()
                data_source, data_target = self.dataloader.load()
                curr_obs = fft_observe(data_source, mask)
                action, prob = self.get_action(curr_obs, mask=mask)
                
                next_obs_last_slice, reward = self.step(action, data_target, copy.deepcopy(mask)) 
                reward_horizon += reward
                v = self.valnet(curr_obs)
                with torch.no_grad():
                    next_obs = torch.concat((curr_obs[:,1:,:,:],next_obs_last_slice),dim=1)
                    vnew  = self.valnet(next_obs)
                    delta = reward + self.gamma * vnew  - v # should check if delta == 0
                self.optimizer_val.zero_grad()
                val_loss = - delta * v
                val_loss.backward()
                self.optimizer_val.step()
                
                self.optimizer_poly.zero_grad()
                poly_loss = - I * delta * torch.log(prob)
                poly_loss.backward()
                self.optimizer_poly.step()
                
                I *= self.gamma
                mask[action] = 1
                print(f'step: {self.steps}, poly_loss: {poly_loss.detach().item():.4f}, val_loss: {val_loss.detach().item():.4f}, reward: {reward.mean().item():.4f}, \n mask sum: {mask.sum().item()}')
                torch.cuda.empty_cache()
            self.reward_per_horizon.append(reward_horizon)
    
    def save(self):
        filename = f'AC1_hist.pt'
        torch.save(
                    {
                        "polynet_weights": self.polynet.state_dict(),
                        "valnet_weights": self.valnet.state_dict(),
                        "reward_per_horizon":self.reward_per_horizon,
                    },
                    self.save_dir + filename,
                )
        
