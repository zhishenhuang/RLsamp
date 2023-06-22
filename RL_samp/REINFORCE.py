from .header import *
from .utils import fft_observe, mask_naiveRand, NRMSE, ssim_uniform
from .reconstructors import unet_solver

##########################################
# REINFORCEMENT
##########################################
class REINFORCE_trainer():
    def __init__(self, dataloader, polynet, 
                  fulldim:int=144,base:int=5,budget:int=13,
                  gamma:float=.8,
                  horizon:int=None,
                  max_trajectories:int=100,
                  reward_scale:float=1,
                  lr=1e-5,
                  save_dir:str='/home/huangz78/rl_samp/',
                  device=torch.device('cpu'),
                  device_alt=torch.device('cpu'),
                  freq_dqn_checkpoint_save:int=10,
                  unet=None,
                  rand_eval_unet=None,
                  lowfreq_eval_unet=None,
                  infostr="",
                  mag_weight:float=1.,
                  guide_epochs:int=0,
                  whitening:bool=False,
                  entropy_reg_scale:float=0):
        
        self.dataloader = dataloader
        self.dataloader.reset()
        
        self.device  = device
        self.device_alt = device_alt
        
        self.polynet    = polynet.to(self.device)
        self.fulldim    = fulldim
        self.base       = int(base)
        self.budget     = int(budget)
        self.gamma      = gamma
        self.reward_scale = reward_scale
        if horizon is None:
            self.horizon = self.budget
        else:
            self.horizon = int(horizon)
        self.optimizer_poly = optim.Adam(self.polynet.parameters(), lr=lr)
        self.max_trajectories = int(max_trajectories)
        
        ## reconstructors
        self.unet = unet
        self.rand_eval_unet    = rand_eval_unet
        self.lowfreq_eval_unet = lowfreq_eval_unet
        
        self.reward_per_horizon = []
        self.save_dir = save_dir
        self.freq_dqn_checkpoint_save = freq_dqn_checkpoint_save
        
        self.steps = 0
        self.epoch = 0
        self.train_hist = {'poly_loss':[], 'poly_grad_norm':[], 
                           'action_prob':[], 'horizon_rewards':[], 
                           'rmse':[], 'recon_samples':[], 'rmse_rand':[], 'rmse_lowfreq':[]}
        
        self.mag_weight   = mag_weight
        self.guide_epochs = max(self.max_trajectories//2,1) if guide_epochs is None else guide_epochs
        self.infostr      = infostr + f'_magweg{mag_weight}_rwd{reward_scale}' if infostr is not None else f'_magweg{mag_weight}_rwd{reward_scale}'
        
        self.whitening = False
        self.entropy_reg_scale = entropy_reg_scale
        self.eps = np.finfo(np.float32).eps.item()
        
    def get_action(self, curr_obs, mask=None):
        '''
        here the mask is 1D
        '''
        if mask is not None:
            assert(len(mask.shape)==1)
        res  = self.polynet(curr_obs,mask) # Jul 5, add mask here as second input
        loc  = torch.argmax(res)
        prob = res.gather(dim=1,index=loc.long().view(-1,1)).squeeze()
        if self.entropy_reg_scale > 0:
            entropy = torch.sum( (-1.) * res * torch.log2(res + .1*self.eps) )
            return loc, prob, entropy
        else:
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
            target_obs_freq, magnitude = fft_observe(target_gt, mask, return_opt='img', action=action, abs_opt=False)
            img_recon = unet_solver(target_obs_freq.to(self.device_alt[0]), self.unet) # target_obs_freq dim: [N, 2, H, W]
            old_nrmse = NRMSE(img_recon,target_gt.to(self.device_alt[0]))

            ### observe target_gt with new freq info
            mask[action] = 1 # incorporate action into mask
            target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            next_obs  = unet_solver(target_obs_freq.to(self.device_alt[0]), self.unet)
            new_nrmse = NRMSE(next_obs,target_gt.to(self.device_alt[0]))

            ## modified on Feb 21, with additional reward given to large magnitude
            reward_extra = max(np.cos(epoch/self.guide_epochs * np.pi/2), 0) * self.mag_weight * magnitude if self.guide_epochs > 0 else torch.tensor(0).to(self.device_alt[0])
            reward = max(old_nrmse - new_nrmse,0) + reward_extra
        
        return next_obs.to(self.device), reward.to(self.device)*self.reward_scale, [next_obs.to(torch.device('cpu')),target_gt.to(torch.device('cpu'))]
    
    def rand_eval(self, target_gt):
        with torch.no_grad():
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=self.budget,roll=False) 
            target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            recon_rand      = unet_solver(target_obs_freq.to(self.device_alt[1]), self.rand_eval_unet)
            res = NRMSE(recon_rand, target_gt.to(self.device_alt[1]))
        return res.detach().item()
    
    def lowfreq_eval(self, target_gt):
        with torch.no_grad():
            mask = mask_naiveRand(self.fulldim,fix=self.base+self.budget,other=0,roll=False) # curr_state
            target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            recon_lowfreq   = unet_solver(target_obs_freq.to(self.device), self.lowfreq_eval_unet)
            res = NRMSE(recon_lowfreq, target_gt.to(self.device))
        return res.detach().item()
          
    def run(self):
#         for trajectory in range(self.max_trajectories):
#             print(f'trajectory [{trajectory+1}/{self.max_trajectories}]')
        
        while (self.dataloader.reset_count-1)//self.dataloader.file_count<self.max_trajectories:
            print(f'epoch [{self.dataloader.reset_count//self.dataloader.file_count +1}/{self.max_trajectories}] file [{self.dataloader.reset_count%self.dataloader.file_count}/{self.dataloader.file_count}] rep [{self.dataloader.rep +1}/{self.dataloader.rep_ubd}] slice [{self.dataloader.slice +1}/{self.dataloader.slice_ubd}]')
            
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False) # curr_state
            
            reward_episode = []
            prob_episode   = []
            if self.entropy_reg_scale > 0:
                entropy_action_distr_episode = []
            
            ## within an episode
            for t in range(self.horizon):
                self.steps += 1
                data_source, data_target = self.dataloader.load()
                curr_obs = fft_observe(data_source, mask, return_opt='img',abs_opt=True)
                curr_obs = curr_obs.to(self.device)
                
                mask = mask.to(self.device)
                if self.entropy_reg_scale > 0:
                    action, prob, entropy = self.get_action(curr_obs, mask=mask)
                    entropy_action_distr_episode.append(entropy)
                else:
                    action, prob = self.get_action(curr_obs, mask=mask)
                prob_episode.append(prob)
                _, reward, recon_pair = self.step(action, data_target, copy.deepcopy(mask), self.epoch) 
                reward_episode.append(reward)
                ### update state
                mask[action] = 1
                
            G = []
            R = 0
            for r in reward_episode[::-1]:
                R = r + self.gamma * R
                G.insert(0, R)
            
            G = torch.tensor(G) 
            if self.whitening:
                G = (G - G.mean()) / (G.std() + .1*self.eps)
            
            p_losses = []
            for prob, R in zip(prob_episode, G):
                p_losses.append(-1. * torch.log(prob) * R)
                
            self.optimizer_poly.zero_grad()
            if self.entropy_reg_scale > 0:
                entropy_regularizer = (-1.)*self.entropy_reg_scale * torch.stack(entropy_action_distr_episode).sum() # want high entropy
                policy_loss = torch.stack(p_losses).sum() + entropy_regularizer
            else:
                policy_loss = torch.stack(p_losses).sum()
            policy_loss.backward()
            self.optimizer_poly.step()

            ### Save training recon samples
            rmse_tmp = NRMSE(recon_pair[0],recon_pair[1])
            self.train_hist['rmse'].append( rmse_tmp )
            print(f'step: {self.steps}, rmse {rmse_tmp}')
            rmse_rand = self.rand_eval(recon_pair[1])
            print(f'step: {self.steps}, rmse_rand {rmse_rand}')
            self.train_hist['rmse_rand'].append( rmse_rand )
            rmse_lowfreq = self.lowfreq_eval(recon_pair[1])
            print(f'step: {self.steps}, rmse_lowfreq {rmse_lowfreq}')
            self.train_hist['rmse_lowfreq'].append( rmse_lowfreq )

            if np.random.rand() <= 0.1:
                recon_pair.append(mask) # save mask as well
                self.train_hist['recon_samples'].append(recon_pair)

            ### For polynet, compute total gradient norm (for logging purposes) and then clip gradients
            grad_norm: torch.Tensor = 0  
            for p in list(filter(lambda p: p.grad is not None, self.polynet.parameters())):
                grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            self.train_hist['poly_grad_norm'].append(grad_norm)
            self.train_hist['poly_loss'].append(policy_loss.detach().item())
            self.train_hist['action_prob'].append(prob.detach().item())           
            self.train_hist['horizon_rewards'].append(G[0].detach().item())            
            print(f'step: {self.steps}, poly_loss: {policy_loss.detach().item():.4f}, reward: {G[0].detach().item():.4f}, \n mask sum: {mask.sum().item()}')
                
            torch.cuda.empty_cache()
            torch.backends.cuda.cufft_plan_cache.clear()
                
            if self.steps % self.freq_dqn_checkpoint_save == 0:
                self.save()
            self.epoch = (self.dataloader.reset_count-1)//self.dataloader.file_count
               
    def save(self):
        filename = f'REINFORCE_hist_{str(datetime.date.today())}_base{self.base}_budget{self.budget}'
        if self.infostr is not None:
            filename = filename + '_' + self.infostr
        filename = filename + '.pt'
        torch.save(
                    {
                        "polynet_weights": self.polynet.state_dict(),
                        "training_record":self.train_hist,
                    },
                    self.save_dir + filename,
                )
        
        print(f'~~ step {self.steps}: hist saved as {filename} at directory {self.save_dir}~~')

class REINFORCE_tester():
    def __init__(self, dataloader, polynet, 
                  fulldim:int=144,base:int=5,budget:int=13,
                  horizon:int=None,
                  reward_scale:float=1,
                  save_dir:str='/home/huangz78/rl_samp/',
                  device=torch.device('cpu'),
                  device_alt=torch.device('cpu'),
                  unet=None,
                  rand_eval_unet=None,
                  lowfreq_eval_unet=None,
                  prob_eval_unet=None,
                  probdistr=None,
                  infostr="",
                  mag_weight:float=1.,):
        
        self.dataloader = dataloader
        self.dataloader.reset()
        
        self.device  = device
        
        self.polynet = polynet.to(self.device)
        self.fulldim = fulldim
        self.base    = int(base)
        self.budget  = int(budget)
        self.reward_scale = reward_scale
        if horizon is None:
            self.horizon = self.budget
        else:
            self.horizon = int(horizon)
        
        ## reconstructors
        self.unet = unet
        self.rand_eval_unet    = rand_eval_unet
        self.lowfreq_eval_unet = lowfreq_eval_unet
        self.prob_eval_unet    = prob_eval_unet
        self.probdistr = probdistr
        
        self.save_dir = save_dir        

        self.test_hist = {'rmse':[], 'recon_samples':[], 'rmse_rand':[], 'rmse_lowfreq':[],'rmse_prob':[],
                          'ssim':[], 'ssim_rand':[], 'ssim_lowfreq':[],'ssim_prob':[]}
        self.guide_epochs = 0
        self.mag_weight   = mag_weight
        self.infostr      = infostr + f'_magweg{mag_weight}_rwd{reward_scale}' if infostr is not None else f'_magweg{mag_weight}_rwd{reward_scale}'
        
        self.eps = np.finfo(np.float32).eps.item()
        
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
            target_obs_freq, magnitude = fft_observe(target_gt, mask, return_opt='img', action=action, abs_opt=False)
            img_recon = unet_solver(target_obs_freq.to(self.device), self.unet) # target_obs_freq dim: [N, 2, H, W]
            old_nrmse = NRMSE(img_recon,target_gt.to(self.device))

            ### observe target_gt with new freq info
            mask[action] = 1 # incorporate action into mask
            target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            next_obs  = unet_solver(target_obs_freq.to(self.device), self.unet)
            new_nrmse = NRMSE(next_obs,target_gt.to(self.device))

            ## modified on Feb 21, with additional reward given to large magnitude
            reward_extra = max(np.cos(epoch/self.guide_epochs * np.pi/2), 0) * self.mag_weight * magnitude if self.guide_epochs > 0 else torch.tensor(0).to(self.device)
            reward = max(old_nrmse - new_nrmse,0) + reward_extra
        
        return next_obs.to(self.device), reward.to(self.device)*self.reward_scale, [next_obs.to(torch.device('cpu')),target_gt.to(torch.device('cpu'))]
    
    def rand_eval(self, target_gt):
        with torch.no_grad():
            mask = mask_naiveRand(self.fulldim,fix=self.base,other=self.budget,roll=False) 
            target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            recon_rand      = unet_solver(target_obs_freq.to(self.device), self.rand_eval_unet)
            nrmse_res = NRMSE(recon_rand, target_gt.to(self.device))
            ssim_res  = ssim_uniform(recon_rand,target_gt.to(self.device),reduction = 'mean')
        return nrmse_res, ssim_res
    
    def lowfreq_eval(self, target_gt):
        with torch.no_grad():
            mask = mask_naiveRand(self.fulldim,fix=self.base+self.budget,other=0,roll=False) # curr_state
            target_obs_freq = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            recon_lowfreq   = unet_solver(target_obs_freq.to(self.device), self.lowfreq_eval_unet)
            nrmse_res = NRMSE(recon_lowfreq, target_gt.to(self.device))
            ssim_res  = ssim_uniform(recon_lowfreq,target_gt.to(self.device),reduction = 'mean')
        return nrmse_res, ssim_res
    
    def mask_given_prob(self,sampdim,probdistr,fix=8,other=16,roll=False):
        '''
        input imgs: [NCHW]
        '''
        fix   = int(fix)
        other = int(other)
        
        fixInds  = np.concatenate((np.arange(0,round(fix//2) ),np.arange(sampdim-1,sampdim-1-round(fix/2),-1)))
        addInds  = np.random.choice(np.arange(sampdim),size=other,replace=False,p=probdistr.numpy())
        maskInds = np.concatenate((fixInds,addInds))
        mask     = np.zeros(sampdim)
        mask[maskInds]= 1
        if roll:
            mask = np.roll(mask,shift=sampdim//2,axis=0)
        return mask
    
    def prob_eval(self, target_gt):
        with torch.no_grad():
            mask = self.mask_given_prob(sampdim=self.fulldim,probdistr=self.probdistr,fix=self.base,other=self.budget) # curr_state
            target_obs_prob = fft_observe(target_gt,mask,return_opt='img', abs_opt=False)
            recon_prob   = unet_solver(target_obs_prob.to(self.device), self.prob_eval_unet)
            nrmse_res = NRMSE(recon_prob, target_gt.to(self.device))
            ssim_res  = ssim_uniform(recon_prob,target_gt.to(self.device),reduction = 'mean')
        return nrmse_res, ssim_res
          
    def run(self):
        with torch.no_grad():
            while (self.dataloader.reset_count-1)//self.dataloader.file_count<1:
                print(f'file [{self.dataloader.reset_count%self.dataloader.file_count}/{self.dataloader.file_count}] rep [{self.dataloader.rep +1}/{self.dataloader.rep_ubd}] slice [{self.dataloader.slice +1}/{self.dataloader.slice_ubd}]')

                mask = mask_naiveRand(self.fulldim,fix=self.base,other=0,roll=False) # curr_state
                save_flag = True if np.random.rand() < 0.4 else False

                for t in range(self.horizon):
                    data_source, data_target = self.dataloader.load()
                    curr_obs = fft_observe(data_source, mask, return_opt='img',abs_opt=True)
                    curr_obs = curr_obs.to(self.device)

                    mask = mask.to(self.device)
                    action, prob = self.get_action(curr_obs, mask=mask)
                    _, reward, recon_pair = self.step(action, data_target, copy.deepcopy(mask)) 
                    ### update state
                    mask[action] = 1
                    if save_flag:
                        recon_pair.append(mask)
                        self.test_hist['recon_samples'].append( recon_pair )

                rmse_rl_currFile      = []
                rmse_rand_currFile    = []
                rmse_lowfreq_currFile = []
                rmse_prob_currFile    = []
                ssim_rl_currFile      = []
                ssim_rand_currFile    = []
                ssim_lowfreq_currFile = []
                ssim_prob_currFile    = []
                x, x_gt = recon_pair[0],recon_pair[1]

                while data_target is not None:
                    rmse_rl_currFile.append( NRMSE(x, x_gt).item() )
                    ssim_rl_currFile.append( ssim_uniform(x,x_gt,reduction = 'mean').item() )

                    rand_res = self.rand_eval(data_target)
                    lowfreq_res = self.lowfreq_eval(data_target)
                    prob_res = self.prob_eval(data_target)
                    
                    rmse_rand_currFile.append( rand_res[0].item() )
                    rmse_lowfreq_currFile.append( lowfreq_res[0].item() )
                    rmse_prob_currFile.append( prob_res[0].item() )
                    ssim_rand_currFile.append( rand_res[1].item() )
                    ssim_lowfreq_currFile.append( lowfreq_res[1].item() )
                    ssim_prob_currFile.append( prob_res[1].item() )

                    _, data_target = self.dataloader.load()
                    if data_target is not None:
                        curr_obs  = fft_observe(data_target,mask,return_opt='img',abs_opt=False)
                        img_recon = unet_solver(curr_obs.to(self.device), self.unet)
                        x , x_gt  = img_recon, data_target.to(self.device)
                        if save_flag:
                            self.test_hist['recon_samples'].append( [x, x_gt, mask] )

                self.test_hist['rmse'].append( sum(rmse_rl_currFile)/len(rmse_rl_currFile) )                
                self.test_hist['rmse_rand'].append( sum(rmse_rand_currFile)/len(rmse_rand_currFile) )
                self.test_hist['rmse_lowfreq'].append( sum(rmse_lowfreq_currFile)/len(rmse_lowfreq_currFile) ) 
                self.test_hist['rmse_prob'].append( sum(rmse_prob_currFile)/len(rmse_prob_currFile) )

                self.test_hist['ssim'].append( sum(ssim_rl_currFile)/len(ssim_rl_currFile) )                
                self.test_hist['ssim_rand'].append( sum(ssim_rand_currFile)/len(ssim_rand_currFile) )
                self.test_hist['ssim_lowfreq'].append( sum(ssim_lowfreq_currFile)/len(ssim_lowfreq_currFile) )  
                self.test_hist['ssim_prob'].append( sum(ssim_prob_currFile)/len(ssim_prob_currFile) )

                self.dataloader.reset()

            self.save()
            
    def save(self):
        filename = f'Test_REINFORCE_hist_{str(datetime.date.today())}_base{self.base}_budget{self.budget}'
        if self.infostr is not None:
            filename = filename + '_' + self.infostr
        filename = filename + '.pt'
        torch.save(
                    {
                        "testing_record":self.test_hist,
                    },
                    self.save_dir + filename,
                )
        
        print(f'~~ hist saved as {filename} at directory {self.save_dir}~~')