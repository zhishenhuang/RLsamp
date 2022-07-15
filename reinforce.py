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

    def reinforce(self):
        score = []
        for i in range(self.traj):
            curr_state = env.reset()
            done = False
            transitions = []
            
            for t in range(self.horizon):
                act_prob = self.model(torch.from_numpy(curr_state).float())
                action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
                prev_state = curr_state
                curr_state, _, done, info = env.step(action)
                transitions.append((prev_state, action, t+1))
                if done:
                    break
            score.append(len(transitions))
            reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,)) 

            batch_Gvals = []
            for i in range(len(transitions)):
                new_Gval=0
                power=0
                for j in range(i,len(transitions)):
                    new_Gval=new_Gval+((self.gamma**power)*reward_batch[j]).numpy()
                    power+=1
                batch_Gvals.append(new_Gval)
            expected_returns_batch=torch.FloatTensor(batch_Gvals)
            
            
            expected_returns_batch /= expected_returns_batch.max()

            state_batch = torch.Tensor([s for (s,a,r) in transitions]) 
            action_batch = torch.Tensor([a for (s,a,r) in transitions]) 

            pred_batch = self.model(state_batch) 
            prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() 
            
            loss = - torch.sum(torch.log(prob_batch) * expected_returns_batch) 
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
