from torch import nn
from torch.nn.init import xavier_uniform, constant
import torch.nn.functional as F

class FFNet(nn.Module):
    """
    Two hidden layer, feed-forward neural network
    """
    def __init__(self,in_size,out_size,width=32):
        super(FFNet, self).__init__()

        # two hidden layers of size width
        self.affine_1 = nn.Linear(in_size, width)
        xavier_uniform(self.affine_1.weight)
        constant(self.affine_1.bias,0.)
        self.affine_2 = nn.Linear(width,width)
        xavier_uniform(self.affine_2.weight)
        constant(self.affine_2.bias,0.)
        self.affine_out = nn.Linear(width, out_size)
        xavier_uniform(self.affine_out.weight)
        constant(self.affine_out.bias,0.)

    def forward(self, x):
        x = F.tanh(self.affine_1(x))
        x = F.tanh(self.affine_2(x))
        x = self.affine_out(x)
        return x