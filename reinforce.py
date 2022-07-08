from header import *
from utils import *

class REINFORCE():
    def __init__(self,model,memory,
                    gamma:float=.5,
                    lr:float=.1,
                    traj:int=500):
        self.model = model
        self.memory = memory
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.traj = traj