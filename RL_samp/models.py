from .header import *

class poly_net(nn.Module):
    def __init__(self, samp_dim=144, in_chans=3, softmax=False):
        super(poly_net, self).__init__()
        self.softmax = softmax
        self.Softmax = nn.Softmax(dim=1)
        self.in_chans  = in_chans
        self.mid_chans = 2 * self.in_chans
#         self.out_chans = 2 * self.mid_chans
        self.samp_dim  = samp_dim
        self.conv1 = nn.Conv2d(in_channels=self.in_chans,    out_channels=self.mid_chans,   kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.mid_chans,   out_channels=2*self.mid_chans, kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=2*self.mid_chans, out_channels=4*self.mid_chans, kernel_size=3,stride=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(1680, 500) 
        self.fc2   = nn.Linear(500, samp_dim)
    
    @property
    def num_param(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
    
    def forward(self, img_input, mask=None):
        # the input mask should be binary
        if mask is not None:
            if len(mask.shape) == 1:
                mask = mask.repeat(img_input.shape[0],1)
            assert(self.samp_dim == mask.shape[1])
            assert(img_input.shape[0] == mask.shape[0])
        else:
            mask = torch.zeros((img_input.shape[0],self.samp_dim))
        x = self.pool(Func.relu(self.conv1(img_input)))
        x = self.pool(Func.relu(self.conv2(x)))
        x = self.pool(Func.relu(self.conv3(x)))
        x = x.view(-1, 1680)
        x = Func.relu(self.fc1(x))
        x = self.fc2(x)
        if self.softmax:
            return self.Softmax(x - 1e10*mask)
        else:
            return x - 1e10*mask
       
        
class val_net(nn.Module):
    def __init__(self, in_chans=3):
        super(val_net, self).__init__()
        self.in_chans  = in_chans
        self.mid_chans = 2 * self.in_chans
        self.conv1 = nn.Conv2d(in_channels=self.in_chans,    out_channels=self.mid_chans,   kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.mid_chans,   out_channels=2*self.mid_chans, kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=2*self.mid_chans, out_channels=4*self.mid_chans, kernel_size=3,stride=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(1680, 500) 
        self.fc2   = nn.Linear(500, 1)
    
    @property
    def num_param(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
    
    def forward(self, img_input):
        x = self.pool(Func.leaky_relu(self.conv1(img_input)))
        x = self.pool(Func.leaky_relu(self.conv2(x)))
        x = self.pool(Func.leaky_relu(self.conv3(x)))
        x = x.view(-1, 1680)
        x = Func.leaky_relu(self.fc1(x))
        x = Func.leaky_relu(self.fc2(x))
        return x