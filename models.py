from header import *

class poly_net(nn.Module):
    def __init__(self, samp_dim=144, in_chans=3, args=True):
        super(poly_net, self).__init__()
        self.in_chans  = in_chans
        self.mid_chans = 2 * self.in_chans
#         self.out_chans = 2 * self.mid_chans
        self.samp_dim  = samp_dim
        self.conv1 = nn.Conv2d(in_channels=self.in_chans,  out_channels=self.mid_chans, kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.mid_chans, out_channels=2*self.mid_chans, kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=2*self.mid_chans, out_channels=4*self.mid_chans, kernel_size=3,stride=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(1680, 500) 
        self.fc2   = nn.Linear(500, samp_dim)
    @property
    def num_param(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
    
    def forward(self, x, mask=None):
        if mask is not None:
            assert(self.samp_dim == mask.shape[1])
        else:
            mask = torch.zeros((1,self.samp_dim))
        x = self.pool(Func.relu(self.conv1(x)))
        x = self.pool(Func.relu(self.conv2(x)))
        x = self.pool(Func.relu(self.conv3(x)))
        x = x.view(-1, 1680)
        x = Func.relu(self.fc1(x))
        x = self.fc2(x)
        return x - 1e10*mask