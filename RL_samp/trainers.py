from .header import *
from .utils import fft_observe, mask_naiveRand, NRMSE
from .reconstructors import sigpy_solver

class DeepQL_trainer():
    def __init__(self,dataloader,policy,
                 episodes:int=10,
                 eps:float=1e-3,
                 fulldim:int=144,
                 base:int=10,
                 budget:int=50,
                 freq_dqn_checkpoint_save:int=10,
                 save_dir:str='/home/huangz78/rl_samp/',
                 ngpu=1,
                 compare=True):
        self.dataloader = dataloader
        self.dataloader.reset()
        
        self.policy   = policy
        self.episodes = episodes
        self.epi      = 0
        self.fulldim  = fulldim
        self.base     = base
        self.budget   = budget
        self.eps      = eps
        self.training_record = {'loss':[], 'grad_norm':[], 'q_values_mean':[], 'q_values_std':[],
                                'horizon_rewards':[],
                                'rmse':[], 'recon_samples':[], 'rmse_rand':[], 'rmse_lowfreq':[],
                                'recon_samples_rand':[]}
        self.steps    = 0
        self.save_dir = save_dir
        self.freq_dqn_checkpoint_save = freq_dqn_checkpoint_save
        self.ngpu     = ngpu
        self.compare  = compare
        
        if self.ngpu > 0:
            self.policy.model = self.policy.model.cuda()
    
    def lowfreq_eval(self, target_gt):
        mask = mask_naiveRand(self.fulldim,fix=self.base+self.budget,other=0,roll=False) # curr_state
        target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
        recon_lowfreq   = sigpy_solver(target_obs_freq, 
                                 L=self.policy.L,max_iter=self.policy.max_iter,solver=self.policy.solver,
                                 heg=target_gt.shape[2],wid=target_gt.shape[3])
        return NRMSE(recon_lowfreq,target_gt)
    
    def train(self):  
        # run training
        while (self.dataloader.reset_count-1)//self.dataloader.file_count<self.episodes:
            print(f'epoch [{self.dataloader.reset_count//self.dataloader.file_count +1}/{self.episodes}] file [{self.dataloader.reset_count%self.dataloader.file_count}/{self.dataloader.file_count}] rep [{self.dataloader.rep +1}/{self.dataloader.rep_ubd}] slice [{self.dataloader.slice +1}/{self.dataloader.slice_ubd}]')
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
                    next_obs, reward, mask_RL, recon_pair_RL = self.policy.step(action, data_target, mask_RL)
#                     print(f'step: {self.steps}, policy.step, mask_RL sum: {mask_RL.sum().item()}')
                    horizon_reward_total += reward
                    self.policy.memory.push(curr_obs, mask, action, next_obs, reward)
                    mask = copy.deepcopy(mask_RL)
#                     print(f'step: {self.steps}, assign, mask sum: {mask.sum().item()}')
    
                    ### compare with random policy
                    if self.compare:
                        action_rand = self.policy.get_rand_action(mask=mask_rand)
                        _, reward_rand, mask_rand, recon_pair_rand = self.policy.step(action_rand, data_target, mask_rand)
                        rmse_lowfreq = self.lowfreq_eval(data_target)
                        
                    ### save recon result for tracking 
                    if mask.sum() == self.budget + self.base:
                        rmse_tmp = NRMSE(recon_pair_RL[0],recon_pair_RL[1])
                        self.training_record['rmse'].append( rmse_tmp.item() )
                        print(f'step: {self.steps}, rmse {rmse_tmp.item()}')
                        if self.compare:
                            rmse_rand = NRMSE(recon_pair_rand[0],recon_pair_rand[1])
                            self.training_record['rmse_rand'].append(rmse_rand.item())
                            self.training_record['rmse_lowfreq'].append(rmse_lowfreq.item())
                            print(f'step: {self.steps}, rmse_rand {rmse_rand.item()}')
                            print(f'step: {self.steps}, rmse_lowfreq {rmse_lowfreq.item()}')
                            
                        if np.random.rand() <= 0.1:
                            recon_pair_RL.append(mask)
                            self.training_record['recon_samples'].append( recon_pair_RL )
                            if self.compare:
                                recon_pair_rand.append(mask_rand)
                                self.training_record['recon_samples_rand'].append( recon_pair_rand )
                
                update_results = self.policy.update_parameters()
                if update_results is not None:
                    for key in ['loss', 'grad_norm','q_values_mean','q_values_std']:
                        if key == 'loss':
                            self.training_record[key].append(update_results[key].detach().item())
                        elif key == 'grad_norm':
                            self.training_record[key].append(update_results[key])
                        else:
                            self.training_record[key].append(update_results[key].item())
                    curr_loss = update_results['loss']
                    if self.compare:
                        print(f'step: {self.steps}, loss: {curr_loss:.4f}, RL reward: {reward.mean().item():.4f}, Rand reward: {reward_rand.mean().item():.4f} \n mask sum: {mask.sum().item()}')
                    else:
                        print(f'step: {self.steps}, loss: {curr_loss:.4f}, RL reward: {reward.mean().item():.4f}, \n mask sum: {mask.sum().item()}')
                    torch.cuda.empty_cache()
                else:
                    print(f'step: {self.steps}, burn in, mask sum: {mask.sum().item()}')
                
                if self.steps % self.freq_dqn_checkpoint_save == 0:
                    self.save()
                if type(self.policy).__name__.lower() == 'ddqn':
                    if self.steps % self.policy.target_net_update_freq == 0:
                        self.policy.target_model.load_state_dict(self.policy.model.state_dict())
                        print(f'  ~~ At step {self.steps}, target_net is updated. ~~')
            self.training_record['horizon_rewards'].append(horizon_reward_total)
            print(f'step: {self.steps}, episode reward: {horizon_reward_total}')
#             self.dataloader.reset()
    
    def save(self):
        filename = f'{type(self.policy).__name__}_doubleQ_{self.policy.double_q_mode}_hist.pt'
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
        self.policy   = policy
        self.epi      = 0
        self.episodes = self.dataloader.files
        self.fulldim  = fulldim # full size of the dimension to be sampled
        self.base     = base
        self.budget   = budget
        self.eps      = eps
        self.test_record = {'loss':[], 'grad_norm':[],'q_values_mean':[],'q_values_std':[], 'rmse':[]}
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
                self.dataloader.train_mode = True
                while mask.sum() < self.budget + self.base:
                    self.steps += 1
                    data_source, data_target = self.dataloader.load()
    #                 epsilon = _get_epsilon(steps_epsilon, self.options)
                    curr_obs = fft_observe(data_source,mask)
                    action   = self.policy.get_action(curr_obs, mask=mask, eps_threshold=self.eps)
                    next_obs, _, mask, _ = self.policy.step(action, data_target, mask)  # inplace mask change

                ### Do reconstruction 
                self.dataloader.reset_iter() # modifying while loop condition
                self.dataloader.batch_size = 1
                self.dataloader.train_mode = False
                
                data_source, data_target = self.dataloader.load()
                while (data_source is not None):                    
                    curr_obs   = fft_observe(data_source, mask, return_opt='freq', roll=True)
                    img_recon  = sigpy_solver( curr_obs, 
                                                 L=self.policy.L,
                                                 max_iter=self.policy.max_iter,
                                                 solver=self.policy.solver,
                                                 heg=curr_obs.shape[2],wid=curr_obs.shape[3] )
                    curr_nrmse = NRMSE(img_recon,data_source)
                    epi_loss    += curr_nrmse * curr_obs.shape[0]
                    slice_count += curr_obs.shape[0]
                    data_source, data_target = self.dataloader.load()

                self.testRec[self.epi] = epi_loss / slice_count
    #                 if self.steps % self.options.target_net_update_freq == 0:
    #                     self.logger.info("Updating target network.")
    #                     self.target_net.load_state_dict(self.policy.state_dict())
                self.dataloader.reset()
    
        return self.testRec

    
##########################################
# actor-critic 1 trainer, debug needed
##########################################
class AC1_trainer():
    def __init__(self, dataloader, polynet, valnet,
                  fulldim:int=144,base:int=5,budget:int=13,
                  gamma:float=.8,
                  horizon:int=None,
                  max_trajectories:int=100,
                  lr:float=.1,
                  init_base:int=10,
                  L:float=5e-3,
                  max_iter:int=100,
                  solver:str='ADMM',
                  save_dir:str='/home/huangz78/rl_samp/',
                  ngpu:int=1,
                  freq_dqn_checkpoint_save:int=10):
        self.dataloader = dataloader
        self.dataloader.reset()
        self.polynet    = polynet.cuda() if ngpu > 0 else polynet
        self.valnet     = valnet.cuda() if ngpu > 0 else valnet
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
        
        self.reward_per_horizon = []
        self.save_dir = save_dir
        self.ngpu = ngpu
        self.freq_dqn_checkpoint_save = freq_dqn_checkpoint_save
        
        self.steps = 0
        self.train_hist = {'poly_loss':[], 'val_loss':[], 'action_prob':[], 'v':[],
                           'poly_grad_norm':[], 'val_grad_norm':[],
                           'horizon_rewards':[], 
                           'rmse':[], 'recon_samples':[]}
        
    def rand_eval(self, target_gt):
        mask = mask_naiveRand(self.fulldim,fix=self.base,other=self.budget,roll=False) # curr_state
        target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
        recon_rand = sigpy_solver(target_obs_freq, 
                                 L=self.L,max_iter=self.max_iter,solver=self.solver,
                                 heg=target_gt.shape[2],wid=target_gt.shape[3])
        return NRMSE(recon_rand,target_gt)
    
    def lowfreq_eval(self, target_gt):
        mask = mask_naiveRand(self.fulldim,fix=self.base+self.budget,other=0,roll=False) # curr_state
        target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
        recon_lowfreq   = sigpy_solver(target_obs_freq, 
                                 L=self.L,max_iter=self.max_iter,solver=self.solver,
                                 heg=target_gt.shape[2],wid=target_gt.shape[3])
        return NRMSE(recon_lowfreq,target_gt)
        
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
        
        return next_obs, reward, [next_obs,target_gt]
        
    def run(self):
#         for trajectory in range(self.max_trajectories):
#             print(f'trajectory [{trajectory+1}/{self.max_trajectories}]')
        while (self.dataloader.reset_count-1)//self.dataloader.file_count<self.max_trajectories:
            print(f'epoch [{self.dataloader.reset_count//self.dataloader.file_count +1}/{self.max_trajectories}] file [{self.dataloader.reset_count%self.dataloader.file_count}/{self.dataloader.file_count}] rep [{self.dataloader.rep +1}/{self.dataloader.rep_ubd}] slice [{self.dataloader.slice +1}/{self.dataloader.slice_ubd}]')
            
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False) # curr_state
            I = 1
            reward_horizon = 0
            for t in range(self.horizon):
                self.steps += 1
                data_source, data_target = self.dataloader.load()
                curr_obs = fft_observe(data_source, mask)
                if self.ngpu > 0:
                    curr_obs = curr_obs.cuda()
                    mask = mask.cuda()
                action, prob = self.get_action(curr_obs, mask=mask)
                
                next_obs_last_slice, reward, recon_pair = self.step(action, data_target, copy.deepcopy(mask)) 
                reward_horizon += reward
                v = self.valnet(curr_obs)
                with torch.no_grad():
                    next_obs = torch.concat((curr_obs[:,1:,:,:],next_obs_last_slice),dim=1)
                    vnew  = self.valnet(next_obs) if t!=self.horizon-1 else 0
                    delta = reward + self.gamma * vnew  - v # should check if delta == 0
                self.optimizer_val.zero_grad()
                breakpoint()
                val_loss = - delta * v     # Sep 19: val_loss is small in magnitude, not sure if this is an issue
                val_loss.backward()
                self.optimizer_val.step()
                
                self.optimizer_poly.zero_grad()
                poly_loss = - I * delta * torch.log(prob)
                poly_loss.backward()
                self.optimizer_poly.step()
                
                I *= self.gamma
                mask[action] = 1
                
                ### Save training recon samples
                if t == self.horizon-1:
                    self.train_hist['rmse'].append( NRMSE(recon_pair[0],recon_pair[1]) )
                    if np.random.rand < 0.2:
                        recon_pair.append(mask)
                        self.train_hist['recon_samples'].append(recon_pair)
                
                ### Compute total gradient norm (for logging purposes) and then clip gradients
                # for polynet
                grad_norm: torch.Tensor = 0  
                for p in list(filter(lambda p: p.grad is not None, self.polynet.parameters())):
                    grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                self.train_hist['poly_grad_norm'].append(grad_norm)
                
                # for valnet
                grad_norm: torch.Tensor = 0  
                for p in list(filter(lambda p: p.grad is not None, self.valnet.parameters())):
                    grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                self.train_hist['val_grad_norm'].append(grad_norm)
                
                self.train_hist['poly_loss'].append(poly_loss.detach().item())
                self.train_hist['val_loss'].append(val_loss.detach().item())
                self.train_hist['action_prob'].append(prob.detach().item())
                self.train_hist['v'].append(v.detach().item())
                
                print(f'step: {self.steps}, poly_loss: {poly_loss.detach().item():.4f}, val_loss: {val_loss.detach().item():.4f}, reward: {reward.mean().item():.4f}, \n mask sum: {mask.sum().item()}')
                torch.cuda.empty_cache()
                
                if self.steps % self.freq_dqn_checkpoint_save == 0:
                    self.save()
                
            self.train_hist['horizon_rewards'].append(reward_horizon)
    
    def save(self):
        filename = f'AC1_hist_base{self.base}_budget{self.budget}.pt'
        torch.save(
                    {
                        "polynet_weights": self.polynet.state_dict(),
                        "valnet_weights": self.valnet.state_dict(),
                        "training_record":self.train_hist,
                    },
                    self.save_dir + filename,
                )
        
class AC1_ET_trainer():
    def __init__(self, dataloader, polynet, valnet,
                  fulldim:int=144,base:int=5,budget:int=13,
                  gamma:float=.8,
                  horizon:int=None,
                  max_trajectories:int=100,
                  lambda_poly:float=.95,
                  lambda_val:float=.95,
                  alpha_poly:float=.3,
                  alpha_val:float=.3,
                  L:float=5e-3,
                  max_iter:int=100,
                  solver:str='ADMM',
                  save_dir:str='/home/huangz78/rl_samp/',
                  ngpu:int=1,
                  freq_dqn_checkpoint_save:int=10):
        self.dataloader = dataloader
        self.dataloader.reset()
        self.polynet    = polynet.cuda() if ngpu > 0 else polynet
        self.valnet     = valnet.cuda()  if ngpu > 0 else valnet
        self.fulldim    = fulldim
        self.base       = int(base)
        self.budget     = int(budget)
        self.gamma      = gamma
        if horizon is None:
            self.horizon = self.budget
        else:
            self.horizon = int(horizon)
        self.alpha_poly  = alpha_poly
        self.alpha_val   = alpha_val
        self.lambda_poly = lambda_poly
        self.lambda_val  = lambda_val
        
        self.optimizer_val    = optim.SGD(self.valnet.parameters(), lr=1)
        self.optimizer_poly   = optim.SGD(self.polynet.parameters(),lr=1)

        self.max_trajectories = int(max_trajectories)
        ## reconstructor parameters
        self.L = L
        self.max_iter = int(max_iter)
        self.solver   = 'ADMM'
        
        self.reward_per_horizon = []
        self.save_dir = save_dir
        self.ngpu     = ngpu
        self.device   = torch.device('cuda') if ngpu > 0 else torch.device('cpu')
        self.freq_dqn_checkpoint_save = freq_dqn_checkpoint_save
        
        self.steps = 0
        self.train_hist = {'poly_loss':[], 'val_loss':[], 'action_prob':[], 'v':[],
                           'poly_grad_norm':[], 'val_grad_norm':[],
                           'horizon_rewards':[], 
                           'rmse':[], 'recon_samples':[], 'rmse_cmp':[]}
        
    def trace_init(self):
        self.ptr = {}
        for key in self.polynet.state_dict().keys():
            self.ptr[key] = torch.zeros_like(self.polynet.state_dict()[key], device=self.device)
        self.vtr = {}
        for key in self.valnet.state_dict().keys():
            self.vtr[key] = torch.zeros_like(self.valnet.state_dict()[key],  device=self.device)

    def extract_poly_grad(self):
        grad = {}
        for key, param in self.polynet.named_parameters():
            grad[key] = param.grad
        return grad
    
    def extract_val_grad(self):
        grad = {}
        for key, param in self.valnet.named_parameters():
            grad[key] = param.grad
        return grad
    
    def ptr_update(self, gradPi, I):
        for key in self.ptr.keys():
            self.ptr[key] = self.gamma * self.lambda_poly* self.ptr[key] + I * gradPi[key]        
    
    def update_polynet(self,delta):
        polynet_tmp = copy.deepcopy(self.polynet.state_dict())
        for key in self.polynet.state_dict().keys():
            self.polynet.state_dict()[key].copy_(polynet_tmp[key] + self.alpha_poly * delta * self.ptr[key]) 
        del polynet_tmp
        
    def vtr_update(self, gradv):
        for key in self.vtr.keys():
            self.vtr[key] = self.gamma * self.lambda_val * self.vtr[key] + gradv[key]
    
    def update_valnet(self,delta):
        valnet_tmp = copy.deepcopy(self.valnet.state_dict())
        for key in self.valnet.state_dict().keys():
            self.valnet.state_dict()[key].copy_( valnet_tmp[key]  + self.alpha_val  * delta * self.vtr[key]) 
        del valnet_tmp
        
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
        
        return next_obs, reward, [next_obs,target_gt]
    
    def rand_eval(self, target_gt):
        mask = mask_naiveRand(self.fulldim,fix=self.base,other=self.budget,roll=False) # curr_state
        target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
        recon_rand = sigpy_solver(target_obs_freq, 
                                 L=self.L,max_iter=self.max_iter,solver=self.solver,
                                 heg=target_gt.shape[2],wid=target_gt.shape[3])
        return NRMSE(recon_rand,target_gt)
    
    def lowfreq_eval(self, target_gt):
        mask = mask_naiveRand(self.fulldim,fix=self.base+self.budget,other=0,roll=False) # curr_state
        target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
        recon_lowfreq   = sigpy_solver(target_obs_freq, 
                                 L=self.L,max_iter=self.max_iter,solver=self.solver,
                                 heg=target_gt.shape[2],wid=target_gt.shape[3])
        return NRMSE(recon_lowfreq,target_gt)
        
    def run(self):
        
        while (self.dataloader.reset_count-1)//self.dataloader.file_count<self.max_trajectories:
            print(f'epoch [{self.dataloader.reset_count//self.dataloader.file_count +1}/{self.max_trajectories}] file [{self.dataloader.reset_count%self.dataloader.file_count}/{self.dataloader.file_count}] rep [{self.dataloader.rep +1}/{self.dataloader.rep_ubd}] slice [{self.dataloader.slice +1}/{self.dataloader.slice_ubd}]')
            self.trace_init()
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False) # curr_state
            I = 1
            reward_horizon = 0
            for t in range(self.horizon):
                self.steps += 1
                data_source, data_target = self.dataloader.load()
                curr_obs = fft_observe(data_source, mask)
                if self.ngpu > 0:
                    curr_obs = curr_obs.cuda()
                    mask = mask.cuda()
                action, prob = self.get_action(curr_obs, mask=mask)
                
                next_obs_last_slice, reward, recon_pair = self.step(action, data_target, copy.deepcopy(mask)) 
                reward_horizon += reward
                v = self.valnet(curr_obs)
                with torch.no_grad():
                    next_obs = torch.concat((curr_obs[:,1:,:,:],next_obs_last_slice.to(self.device)),dim=1)
                    vnew  = self.valnet(next_obs) if t<self.horizon-1 else 0
                    delta = reward + self.gamma * vnew  - v # should check if delta == 0
                    print(f'step {self.steps}, delta {delta.item()}')
                self.optimizer_val.zero_grad()
                val_loss = -v     # Sep 25: val_loss keeps decreasing without going up when new file of images is loaded
                val_loss.backward()
                vgrad = self.extract_val_grad()
                self.vtr_update(vgrad)
                self.update_valnet(delta.mean().squeeze()) # delta is a [1,1] tensor
                
                self.optimizer_poly.zero_grad()
                poly_loss = -torch.log(prob)
                poly_loss.backward()
                pgrad = self.extract_poly_grad()
                self.ptr_update(pgrad,I)
                self.update_polynet(delta.mean().squeeze())
                
                I *= self.gamma
                mask[action] = 1
                
                ### Save training recon samples
                if t == self.horizon-1:
                    rmse_tmp = NRMSE(recon_pair[0],recon_pair[1])
                    self.train_hist['rmse'].append( rmse_tmp )
                    print(f'step: {self.steps}, rmse {rmse_tmp}')
                    rmse_rand = self.rand_eval(recon_pair[1])
                    print(f'step: {self.steps}, rmse_rand {rmse_rand}')
                    rmse_lowfreq = self.lowfreq_eval(recon_pair[1])
                    print(f'step: {self.steps}, rmse_lowfreq {rmse_lowfreq}')
                    self.train_hist['rmse_cmp'].append([rmse_tmp, rmse_rand, rmse_lowfreq])
                    if np.random.rand() <= 0.1:
                        recon_pair.append(mask) # save mask as well
                        self.train_hist['recon_samples'].append(recon_pair)
                        
                        
                ### Compute total gradient norm (for logging purposes) and then clip gradients
                # for polynet
                grad_norm: torch.Tensor = 0  
                for p in list(filter(lambda p: p.grad is not None, self.polynet.parameters())):
                    grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                self.train_hist['poly_grad_norm'].append(grad_norm)
                
                # for valnet
                grad_norm: torch.Tensor = 0  
                for p in list(filter(lambda p: p.grad is not None, self.valnet.parameters())):
                    grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                self.train_hist['val_grad_norm'].append(grad_norm)
                
                self.train_hist['poly_loss'].append(poly_loss.detach().item())
                self.train_hist['val_loss'].append(val_loss.detach().item())
                self.train_hist['action_prob'].append(prob.detach().item())
                self.train_hist['v'].append(v.detach().item())
                
                print(f'step: {self.steps}, poly_loss: {poly_loss.detach().item():.4f}, val_loss: {val_loss.detach().item():.4f}, reward: {reward.mean().item():.4f}, \n mask sum: {mask.sum().item()}')
                torch.cuda.empty_cache()
                
                if self.steps % self.freq_dqn_checkpoint_save == 0:
                    self.save()
            self.train_hist['horizon_rewards'].append(reward_horizon)
    
    def save(self):
        filename = f'AC1_ET_hist_base{self.base}_budget{self.budget}.pt'
        torch.save(
                    {
                        "polynet_weights": self.polynet.state_dict(),
                        "valnet_weights" : self.valnet.state_dict(),
                        "training_record": self.train_hist,
                    },
                    self.save_dir + filename,
                )
        print(f'~~ step {self.steps}: hist saved as {filename} at directory {self.save_dir}~~')