from .header import *
from .utils import fft_observe, mask_naiveRand, NRMSE, ssim_uniform
from .reconstructors import sigpy_solver, unet_solver

class DeepQL_trainer():
    '''
    mask are always assumed to be UNROLLED!
    '''
    def __init__(self,dataloader,policy,
                 episodes:int=10,
                 eps:float=1e-3,
                 fulldim:int=144,
                 base:int=10,
                 budget:int=50,
                 freq_dqn_checkpoint_save:int=10,
                 save_dir:str='/home/huangz78/rl_samp/',
                 compare=True,
                 rand_eval_unet=None,
                 lowfreq_eval_unet=None,
                 infostr=None,
                 device=torch.device('cpu')):
        self.dataloader = dataloader
        self.dataloader.reset()
        
        self.policy   = policy
        self.episodes = episodes
        self.episode  = 0
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
        
        self.device   = device
        self.compare  = compare
        
        self.rand_eval_unet = rand_eval_unet
        self.lowfreq_eval_unet = lowfreq_eval_unet
        self.infostr  = infostr
        
        self.policy.model = self.policy.model.to(self.device)
            
    def rand_eval(self, target_gt):
        mask = mask_naiveRand(self.fulldim,fix=self.base,other=self.budget,roll=False) 
        reconstructor   = self.policy.unet if self.rand_eval_unet is None else self.rand_eval_unet
        target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
        recon_rand      = unet_solver(target_obs_freq.to(self.device), reconstructor)
        return NRMSE(recon_rand, target_gt.to(self.device))
    
    def lowfreq_eval(self, target_gt):
        mask = mask_naiveRand(self.fulldim,fix=self.base+self.budget,other=0,roll=False) # curr_state
#         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
#         recon_lowfreq   = sigpy_solver(target_obs_freq, 
#                                  L=self.policy.L,max_iter=self.policy.max_iter,solver=self.policy.solver,
#                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
        reconstructor   = self.policy.unet if self.lowfreq_eval_unet is None else self.lowfreq_eval_unet
        target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
        recon_lowfreq   = unet_solver(target_obs_freq.to(self.device), reconstructor)
        return NRMSE(recon_lowfreq, target_gt.to(self.device))
    
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
#                 mask_rand = copy.deepcopy(mask)
#                 epsilon = _get_epsilon(steps_epsilon, self.options)
                curr_obs = fft_observe(data_source,mask_RL,return_opt='img',abs_opt=True) # [N, t_backtrack, H, W]
                with torch.no_grad():
                    action = self.policy.get_action(curr_obs, mask=mask_RL, eps_threshold=self.eps)
#                     print(f'step {self.steps}, action {action}')
                    next_obs, reward, mask_RL, recon_pair_RL = self.policy.step(action, data_target, mask_RL, self.episode) # modified, Dec29
#                     print(f'step: {self.steps}, policy.step, mask_RL sum: {mask_RL.sum().item()}')
                    horizon_reward_total += reward.item()
                    self.policy.memory.push(curr_obs, mask, action, next_obs, reward)
                    mask = copy.deepcopy(mask_RL)
                        
                    ### save recon result for tracking 
                    if mask.sum() == self.budget + self.base:
                        rmse_tmp = NRMSE(recon_pair_RL[0],recon_pair_RL[1])
                        self.training_record['rmse'].append( rmse_tmp.item() )
                        print(f'step: {self.steps}, rmse {rmse_tmp.item()}')
                        if self.compare:
                            rmse_lowfreq = self.lowfreq_eval(data_target)
                            rmse_rand    = self.rand_eval(data_target)
                            
#                             rmse_rand = NRMSE(recon_pair_rand[0],recon_pair_rand[1])
                            self.training_record['rmse_rand'].append(rmse_rand.item())
                            self.training_record['rmse_lowfreq'].append(rmse_lowfreq.item())
                            print(f'step: {self.steps}, rmse_rand {rmse_rand.item()}')
                            print(f'step: {self.steps}, rmse_lowfreq {rmse_lowfreq.item()}')
                            
                        if np.random.rand() <= 0.1:
                            recon_pair_RL.append(mask)
                            self.training_record['recon_samples'].append( recon_pair_RL )
                            
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
                    print(f'step: {self.steps}, loss: {curr_loss:.4f}, RL reward: {reward:.4f}, \n mask sum: {mask.sum().item()}') # reward.mean().item()
                    torch.cuda.empty_cache()
                else:
                    print(f'step: {self.steps}, burn in, mask sum: {mask.sum().item()}')
                
                if self.steps % self.freq_dqn_checkpoint_save == 0:
                    self.save()
                if type(self.policy).__name__.lower() == 'ddqn':
                    if self.steps % self.policy.target_net_update_freq == 0:
                        self.policy.target_model.load_state_dict(self.policy.model.state_dict())
                        print(f'  ~~ At step {self.steps}, target_net is updated. ~~')
                self.episode = (self.dataloader.reset_count-1)//self.dataloader.file_count
                
            self.training_record['horizon_rewards'].append(horizon_reward_total)
            print(f'step: {self.steps}, episode reward: {horizon_reward_total}')
#             self.dataloader.reset()
    
    def save(self):
        filename = f'{type(self.policy).__name__}_doubleQ_{self.policy.double_q_mode}_ba{self.base}_bu{self.budget}_hist_{str(datetime.date.today())}'
        if self.infostr is not None:
            filename = filename + '_' + self.infostr
        filename = filename + '.pt'
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
        
class DeepQL_tester():
    def __init__(self,dataloader,policy,
                 eps:float=1e-3,
                 fulldim:int=144,
                 base:int=10,
                 budget:int=50,
                 save_dir:str='/home/huangz78/rl_samp/',
                 compare=True,
                 rand_eval_unet=None,
                 lowfreq_eval_unet=None,
                 infostr=None,
                 device=torch.device('cpu')
                 ):
        self.dataloader = dataloader
        self.dataloader.reset()
        
        self.policy   = policy
        self.fulldim  = fulldim
        self.base     = base
        self.budget   = budget
        self.eps      = eps
        self.testing_record = {'rmse':[], 'rmse_rand':[], 'rmse_lowfreq':[],
                               'ssim':[], 'ssim_rand':[], 'ssim_lowfreq':[],
                               'recon_samples':[], 'recon_samples_rand':[]}
        self.save_dir = save_dir
        
        self.device   = device
        self.compare  = compare
        
        self.rand_eval_unet = rand_eval_unet
        self.lowfreq_eval_unet = lowfreq_eval_unet
        self.infostr  = infostr
        
        self.policy.model = self.policy.model.to(self.device)
    
    def rand_eval(self, target_gt):
        mask = mask_naiveRand(self.fulldim,fix=self.base,other=self.budget,roll=False) 
        reconstructor   = self.policy.unet if self.rand_eval_unet is None else self.rand_eval_unet
        target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
        recon_rand      = unet_solver(target_obs_freq.to(self.device), reconstructor)
        nrmse_res = NRMSE(recon_rand, target_gt.to(self.device))
        ssim_res  = ssim_uniform(recon_rand,target_gt.to(self.device),reduction = 'mean')
        return nrmse_res, ssim_res
    
    def lowfreq_eval(self, target_gt):
        mask = mask_naiveRand(self.fulldim,fix=self.base+self.budget,other=0,roll=False) # curr_state
        reconstructor   = self.policy.unet if self.lowfreq_eval_unet is None else self.lowfreq_eval_unet
        target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
        recon_lowfreq   = unet_solver(target_obs_freq.to(self.device), reconstructor)
        
        nrmse_res = NRMSE(recon_lowfreq, target_gt.to(self.device))
        ssim_res  = ssim_uniform(recon_lowfreq,target_gt.to(self.device),reduction = 'mean')
        return nrmse_res, ssim_res
    
    def save(self):
        filename = f'EVAL_{type(self.policy).__name__}_doubleQ_{self.policy.double_q_mode}_ba{self.base}_bu{self.budget}_{str(datetime.date.today())}'
        if self.infostr is not None:
            filename = filename + '_' + self.infostr
        filename = filename + '.pt'
        torch.save({
                        "testing_record":self.testing_record,
                    },
                    self.save_dir + filename )
        
        print(f' testing history saved as {self.save_dir + filename}')
    
    def test(self):  
        # run training
        while (self.dataloader.reset_count-1)//self.dataloader.file_count<1:
            print(f'file [{self.dataloader.reset_count%self.dataloader.file_count}/{self.dataloader.file_count}] rep [{self.dataloader.rep +1}/{self.dataloader.rep_ubd}] slice [{self.dataloader.slice +1}/{self.dataloader.slice_ubd}]')
            
            save_flag = True if np.random.rand() < 0.4 else False
            
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False)   
            # one mask at a time, start with a low frequency mask
            while mask.sum() < self.budget + self.base:

                data_source, data_target = self.dataloader.load()
                mask_RL  = copy.deepcopy(mask)

                curr_obs = fft_observe(data_source,mask_RL,return_opt='img',abs_opt=True) # [N, t_backtrack, H, W]
                with torch.no_grad():
                    action = self.policy.get_action(curr_obs, mask=mask_RL, eps_threshold=self.eps)
                    next_obs, reward, mask_RL, recon_pair_RL = self.policy.step(action, data_target, mask_RL) # modified, Dec29
                    mask = copy.deepcopy(mask_RL)
                    
                    if save_flag:
                        recon_pair_RL.append(mask)
                        self.testing_record['recon_samples'].append( recon_pair_RL )
                    
                    ### save recon result for tracking                  
                    if mask.sum() == self.budget + self.base:
                        rmse_rl_currFile      = []
                        rmse_rand_currFile    = []
                        rmse_lowfreq_currFile = []
                        ssim_rl_currFile      = []
                        ssim_rand_currFile    = []
                        ssim_lowfreq_currFile = []
                        x, x_gt = recon_pair_RL[0],recon_pair_RL[1]
                        while data_target is not None:
                            rmse_rl_currFile.append( NRMSE(x, x_gt).item() )
                            ssim_rl_currFile.append( ssim_uniform(x,x_gt.to(self.device),reduction = 'mean').item() )
                            
                            rand_res = self.rand_eval(data_target)
                            lowfreq_res = self.lowfreq_eval(data_target)
                            
                            rmse_rand_currFile.append( rand_res[0].item() )
                            rmse_lowfreq_currFile.append( lowfreq_res[0].item() )
                            ssim_rand_currFile.append( rand_res[1].item() )
                            ssim_lowfreq_currFile.append( lowfreq_res[1].item() )
                            
                            _, data_target = self.dataloader.load()
                            if data_target is not None:
                                curr_obs  = fft_observe(data_target,mask,return_opt='img',abs_opt=False)
                                img_recon = unet_solver(curr_obs.to(self.policy.device), self.policy.unet)
                                x , x_gt  = img_recon, data_target.to(self.policy.device)
                                if save_flag:
                                    self.testing_record['recon_samples'].append( [x, x_gt, mask] )
                            
            self.testing_record['rmse'].append( sum(rmse_rl_currFile)/len(rmse_rl_currFile) )                
            self.testing_record['rmse_rand'].append( sum(rmse_rand_currFile)/len(rmse_rand_currFile) )
            self.testing_record['rmse_lowfreq'].append( sum(rmse_lowfreq_currFile)/len(rmse_lowfreq_currFile) )  
            
            self.testing_record['ssim'].append( sum(ssim_rl_currFile)/len(ssim_rl_currFile) )                
            self.testing_record['ssim_rand'].append( sum(ssim_rand_currFile)/len(ssim_rand_currFile) )
            self.testing_record['ssim_lowfreq'].append( sum(ssim_lowfreq_currFile)/len(ssim_lowfreq_currFile) )  
            self.dataloader.reset()
               
        self.save()
        print(' ~~ Testing Evaluation is completed. ~~')
    
##########################################
# actor-critic 1 trainer, debug needed
##########################################
# class AC1_trainer():
#     def __init__(self, dataloader, polynet, valnet,
#                   fulldim:int=144,base:int=5,budget:int=13,
#                   gamma:float=.8,
#                   horizon:int=None,
#                   max_trajectories:int=100,
#                   lr:float=.1,
#                   init_base:int=10,
#                   L:float=5e-3,
#                   max_iter:int=100,
#                   solver:str='ADMM',
#                   save_dir:str='/home/huangz78/rl_samp/',
#                   device=torch.device('cpu'),
#                   freq_dqn_checkpoint_save:int=10):
#         self.dataloader = dataloader
#         self.dataloader.reset()
#         self.device     = device
#         self.polynet    = polynet.to(self.device)
#         self.valnet     = valnet.to(self.device)
#         self.fulldim    = fulldim
#         self.base       = int(base)
#         self.budget     = int(budget)
#         self.gamma      = gamma
#         if horizon is None:
#             self.horizon = self.budget
#         else:
#             self.horizon = int(horizon)
#         self.lr         = lr
#         self.optimizer_val  = optim.Adam(self.valnet.parameters(),  lr=lr)
#         self.optimizer_poly = optim.Adam(self.polynet.parameters(), lr=lr)
#         self.init_base  = int(init_base)
#         self.max_trajectories = int(max_trajectories)
#         ## reconstructor parameters
#         self.L = L
#         self.max_iter = int(max_iter)
#         self.solver = 'ADMM'
        
#         self.reward_per_horizon = []
#         self.save_dir = save_dir
#         self.ngpu = ngpu
#         self.freq_dqn_checkpoint_save = freq_dqn_checkpoint_save
        
#         self.steps = 0
#         self.train_hist = {'poly_loss':[], 'val_loss':[], 'action_prob':[], 'v':[],
#                            'poly_grad_norm':[], 'val_grad_norm':[],
#                            'horizon_rewards':[], 
#                            'rmse':[], 'recon_samples':[]}
        
#     def rand_eval(self, target_gt):
#         mask = mask_naiveRand(self.fulldim,fix=self.base,other=self.budget,roll=False) # curr_state
#         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
#         recon_rand = sigpy_solver(target_obs_freq, 
#                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
#                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
#         return NRMSE(recon_rand,target_gt)
    
#     def lowfreq_eval(self, target_gt):
#         mask = mask_naiveRand(self.fulldim,fix=self.base+self.budget,other=0,roll=False) # curr_state
#         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
#         recon_lowfreq   = sigpy_solver(target_obs_freq, 
#                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
#                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
#         return NRMSE(recon_lowfreq,target_gt)
        
#     def get_action(self, curr_obs, mask=None):
#         '''
#         here the mask is 1D
#         '''
#         if mask is not None:
#             assert(len(mask.shape)==1)
#         res  = self.polynet(curr_obs,mask) # Jul 5, add mask here as second input
#         loc  = torch.argmax(res)
#         prob = res.gather(dim=1,index=loc.long().view(-1,1)).squeeze()
#         return loc, prob
        
#     def step(self, action, target_gt, mask):
#         '''
#         action: [1] TODO: adding multiple lines at a time
#         mask:[W]
#         data:[N1HW]
#         target:[N1HW]
#         '''
#         ### observe target_gt with old freq info
#         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
#         img_recon = sigpy_solver(target_obs_freq, 
#                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
#                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
#         old_nrmse = NRMSE(img_recon,target_gt)
        
#         ### observe target_gt with new freq info
#         mask[action] = 1 # incorporate action into mask
#         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
#         next_obs  = sigpy_solver(target_obs_freq, 
#                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
#                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
#         new_nrmse = NRMSE(next_obs,target_gt)
#         reward    = old_nrmse - new_nrmse
        
#         return next_obs, reward, [next_obs,target_gt]
        
#     def run(self):
# #         for trajectory in range(self.max_trajectories):
# #             print(f'trajectory [{trajectory+1}/{self.max_trajectories}]')
#         while (self.dataloader.reset_count-1)//self.dataloader.file_count<self.max_trajectories:
#             print(f'epoch [{self.dataloader.reset_count//self.dataloader.file_count +1}/{self.max_trajectories}] file [{self.dataloader.reset_count%self.dataloader.file_count}/{self.dataloader.file_count}] rep [{self.dataloader.rep +1}/{self.dataloader.rep_ubd}] slice [{self.dataloader.slice +1}/{self.dataloader.slice_ubd}]')
            
#             mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False) # curr_state
#             I = 1
#             reward_horizon = 0
#             for t in range(self.horizon):
#                 self.steps += 1
#                 data_source, data_target = self.dataloader.load()
#                 curr_obs = fft_observe(data_source, mask)
                
#                 curr_obs = curr_obs.to(self.device)
#                 mask = mask.to(self.device)
#                 action, prob = self.get_action(curr_obs, mask=mask)
                
#                 next_obs_last_slice, reward, recon_pair = self.step(action, data_target, copy.deepcopy(mask)) 
#                 reward_horizon += reward
#                 v = self.valnet(curr_obs)
#                 with torch.no_grad():
#                     next_obs = torch.concat((curr_obs[:,1:,:,:],next_obs_last_slice),dim=1)
#                     vnew  = self.valnet(next_obs) if t!=self.horizon-1 else 0
#                     delta = reward + self.gamma * vnew  - v # should check if delta == 0
#                 self.optimizer_val.zero_grad()
#                 val_loss = - delta * v     # Sep 19: val_loss is small in magnitude, not sure if this is an issue
#                 val_loss.backward()
#                 self.optimizer_val.step()
                
#                 self.optimizer_poly.zero_grad()
#                 poly_loss = - I * delta * torch.log(prob)
#                 poly_loss.backward()
#                 self.optimizer_poly.step()
                
#                 I *= self.gamma
#                 mask[action] = 1
                
#                 ### Save training recon samples
#                 if t == self.horizon-1:
#                     self.train_hist['rmse'].append( NRMSE(recon_pair[0],recon_pair[1]) )
#                     if np.random.rand < 0.2:
#                         recon_pair.append(mask)
#                         self.train_hist['recon_samples'].append(recon_pair)
                
#                 ### Compute total gradient norm (for logging purposes) and then clip gradients
#                 # for polynet
#                 grad_norm: torch.Tensor = 0  
#                 for p in list(filter(lambda p: p.grad is not None, self.polynet.parameters())):
#                     grad_norm += p.grad.data.norm(2).item() ** 2
#                 grad_norm = grad_norm ** 0.5
#                 self.train_hist['poly_grad_norm'].append(grad_norm)
                
#                 # for valnet
#                 grad_norm: torch.Tensor = 0  
#                 for p in list(filter(lambda p: p.grad is not None, self.valnet.parameters())):
#                     grad_norm += p.grad.data.norm(2).item() ** 2
#                 grad_norm = grad_norm ** 0.5
#                 self.train_hist['val_grad_norm'].append(grad_norm)
                
#                 self.train_hist['poly_loss'].append(poly_loss.detach().item())
#                 self.train_hist['val_loss'].append(val_loss.detach().item())
#                 self.train_hist['action_prob'].append(prob.detach().item())
#                 self.train_hist['v'].append(v.detach().item())
                
#                 print(f'step: {self.steps}, poly_loss: {poly_loss.detach().item():.4f}, val_loss: {val_loss.detach().item():.4f}, reward: {reward.mean().item():.4f}, \n mask sum: {mask.sum().item()}')
#                 torch.cuda.empty_cache()
                
#                 if self.steps % self.freq_dqn_checkpoint_save == 0:
#                     self.save()
                
#             self.train_hist['horizon_rewards'].append(reward_horizon)
    
#     def save(self):
#         filename = f'AC1_hist_base{self.base}_budget{self.budget}.pt'
#         torch.save(
#                     {
#                         "polynet_weights": self.polynet.state_dict(),
#                         "valnet_weights": self.valnet.state_dict(),
#                         "training_record":self.train_hist,
#                     },
#                     self.save_dir + filename,
#                 )
        
class AC1_ET_trainer():
    def __init__(self, dataloader, polynet, valnet,
                  fulldim:int=144,base:int=5,budget:int=13,
                  gamma:float=.8,
                  horizon:int=None,
                  max_trajectories:int=100,
                  lambda_poly:float=.95,
                  lambda_val:float=.95,
                  alpha_poly:float=.3,
                  alpha_val:float=1e-2,
                  L:float=5e-3,
                  max_iter:int=100,
                  solver:str='ADMM',
                  reward_scale:float=1,
                  save_dir:str='/home/huangz78/rl_samp/',
                  device=torch.device('cpu'),
                  device_alt=torch.device('cpu'),
                  freq_dqn_checkpoint_save:int=10,
                  unet=None,
                  rand_eval_unet=None,
                  lowfreq_eval_unet=None,
                  infostr="",
                  mag_weight:float=1.,
                  guide_epochs:int=None):
        self.dataloader = dataloader
        self.dataloader.reset()
        
        self.device  = device
        self.device_alt = device_alt
        self.polynet = polynet.to(self.device)
        self.valnet  = valnet.to(self.device)
        self.fulldim = fulldim
        self.base    = int(base)
        self.budget  = int(budget)
        self.gamma   = gamma
        self.reward_scale = reward_scale
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
        self.freq_dqn_checkpoint_save = freq_dqn_checkpoint_save
        
        self.episode = 0
        self.steps = 0
        self.train_hist = {'poly_loss':[], 'val_loss':[], 'action_prob':[], 'v':[],
                           'poly_grad_norm':[], 'val_grad_norm':[],
                           'horizon_rewards':[], 
                           'rmse':[], 'recon_samples':[], 'rmse_rand':[], 'rmse_lowfreq':[]}
        self.unet = unet
        self.rand_eval_unet    = rand_eval_unet
        self.lowfreq_eval_unet = lowfreq_eval_unet
        
        self.mag_weight   = mag_weight
        self.guide_epochs = max(self.max_trajectories//2,1) if guide_epochs is None else guide_epochs
        self.infostr      = infostr + f'_magweg{mag_weight}_rwd{reward_scale}_val_{self.valnet.scale}'
        
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
        
    def step(self, action, target_gt, mask, epoch=0):
        '''
        action: [1] TODO: adding multiple lines at a time
        mask:[W]
        data:[N1HW]
        target:[N1HW]
        '''
        with torch.no_grad():
            ### observe target_gt with old freq info
    #         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
    #         img_recon = sigpy_solver(target_obs_freq, 
    #                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
    #                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
            target_obs_freq, magnitude = fft_observe(target_gt, mask, return_opt='img', action=action, abs_opt=False)
            img_recon = unet_solver(target_obs_freq.to(self.device_alt[0]), self.unet) # target_obs_freq dim: [N, 2, H, W]
            old_nrmse = NRMSE(img_recon,target_gt.to(self.device_alt[0]))

            ### observe target_gt with new freq info
            mask[action] = 1 # incorporate action into mask
    #         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
    #         next_obs  = sigpy_solver(target_obs_freq, 
    #                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
    #                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
            target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            next_obs  = unet_solver(target_obs_freq.to(self.device_alt[0]), self.unet)
            new_nrmse = NRMSE(next_obs,target_gt.to(self.device_alt[0]))

    #         reward    = old_nrmse - new_nrmse
            ## modified on Feb 21, with additional reward given to large magnitude
            reward_extra = max(np.cos(epoch/self.guide_epochs * np.pi/2), 0) * self.mag_weight * magnitude if self.guide_epochs > 0 else torch.tensor(0).to(self.device_alt[0])
            reward = max(old_nrmse - new_nrmse,0) + reward_extra
        
        return next_obs.to(self.device), reward.to(self.device)*self.reward_scale, [next_obs.to(torch.device('cpu')),target_gt.to(torch.device('cpu'))]
    
    def rand_eval(self, target_gt):
        with torch.no_grad():
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=self.budget,roll=False) 
            target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            recon_rand      = unet_solver(target_obs_freq.to(self.device_alt[1]), self.rand_eval_unet)
    #         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
    #         recon_rand = sigpy_solver(target_obs_freq, 
    #                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
    #                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
            res = NRMSE(recon_rand, target_gt.to(self.device_alt[1]))
        return res.detach().item()
    
    def lowfreq_eval(self, target_gt):
        with torch.no_grad():
            mask = mask_naiveRand(self.fulldim,fix=self.base+self.budget,other=0,roll=False) # curr_state
            target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            recon_lowfreq   = unet_solver(target_obs_freq.to(self.device), self.lowfreq_eval_unet)
    #         target_obs_freq = fft_observe(target_gt,mask,return_opt='freq',roll=True) # roll=True because using sigpy
    #         recon_lowfreq   = sigpy_solver(target_obs_freq, 
    #                                  L=self.L,max_iter=self.max_iter,solver=self.solver,
    #                                  heg=target_gt.shape[2],wid=target_gt.shape[3])
            res = NRMSE(recon_lowfreq, target_gt.to(self.device))
        return res.detach().item()
        
    def run(self):
        
        while (self.dataloader.reset_count-1)//self.dataloader.file_count<self.max_trajectories:
            print(f'epoch [{self.dataloader.reset_count//self.dataloader.file_count +1}/{self.max_trajectories}] file [{self.dataloader.reset_count%self.dataloader.file_count}/{self.dataloader.file_count}] rep [{self.dataloader.rep +1}/{self.dataloader.rep_ubd}] slice [{self.dataloader.slice +1}/{self.dataloader.slice_ubd}]')
            
            self.trace_init()
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False) # curr_state
            I = 1
            reward_horizon = 0.
            for t in range(self.horizon):
                self.steps += 1
                data_source, data_target = self.dataloader.load()
#                 curr_obs = fft_observe(data_source, mask)
                curr_obs = fft_observe(data_source, mask, return_opt='img',abs_opt=True)
                curr_obs = curr_obs.to(self.device)
                mask = mask.to(self.device)
                action, prob = self.get_action(curr_obs, mask=mask)
                
                next_obs_last_slice, reward, recon_pair = self.step(action, data_target, copy.deepcopy(mask), self.episode) 
                reward_horizon += reward
                v = self.valnet(curr_obs)
                with torch.no_grad():
                    next_obs = torch.concat((curr_obs[:,1:,:,:],next_obs_last_slice),dim=1)
                    vnew  = self.valnet(next_obs) if t<self.horizon-1 else 0
                    delta = reward + self.gamma * vnew  - v # should check if delta == 0
                    print(f'step {self.steps}, delta {delta.item()}')
                self.optimizer_val.zero_grad()
                val_loss = v     # Feb 27: removed minus sign
                val_loss.backward()
                vgrad = self.extract_val_grad()
                self.vtr_update(vgrad)
                self.update_valnet(delta.mean().squeeze()) # delta is a [1,1] tensor
                
                self.optimizer_poly.zero_grad()
                poly_loss = torch.log(prob) # Feb 27: removed minus sign
                poly_loss.backward()
                pgrad = self.extract_poly_grad()
                self.ptr_update(pgrad,I)
                self.update_polynet(delta.mean().squeeze())
                
                I *= self.gamma
                mask[action] = 1
                
                ### Save training recon samples
                save_flag = True if np.random.rand() < 0.1 else False
                if t == self.horizon-1:
                    rmse_tmp = NRMSE(recon_pair[0],recon_pair[1])
                    self.train_hist['rmse'].append( rmse_tmp )
                    print(f'step: {self.steps}, rmse {rmse_tmp}')
                    rmse_rand = self.rand_eval(recon_pair[1])
                    print(f'step: {self.steps}, rmse_rand {rmse_rand}')
                    self.train_hist['rmse_rand'].append( rmse_rand )
                    rmse_lowfreq = self.lowfreq_eval(recon_pair[1])
                    print(f'step: {self.steps}, rmse_lowfreq {rmse_lowfreq}')
                    self.train_hist['rmse_lowfreq'].append( rmse_lowfreq )
                    
                    if save_flag: # np.random.rand() <= 0.1:
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
                
                print(f'step: {self.steps}, poly_loss: {poly_loss.detach().item():.4f}, val_loss: {val_loss.detach().item():.4f}, reward: {reward.item():.4f}, \n mask sum: {mask.sum().item()}') # reward.mean().item()
                
                torch.cuda.empty_cache()
                torch.backends.cuda.cufft_plan_cache.clear()
                
                if self.steps % self.freq_dqn_checkpoint_save == 0:
                    self.save()
                self.episode = (self.dataloader.reset_count-1)//self.dataloader.file_count
            self.train_hist['horizon_rewards'].append(reward_horizon.detach().item())
    
    def save(self):
        filename = f'AC1_ET_hist_base{self.base}_budget{self.budget}_{str(datetime.date.today())}_{self.infostr}.pt'
        torch.save(
                    {
                        "polynet_weights": self.polynet.state_dict(),
                        "valnet_weights" : self.valnet.state_dict(),
                        "training_record": self.train_hist,
                    },
                    self.save_dir + filename,
                )
        print(f'~~ step {self.steps}: hist saved as {filename} at directory {self.save_dir}~~')