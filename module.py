import torch
import torch.nn as nn


# dilated causal convolution
class CausalConv1d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 dilation=1, 
                 **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, 
                              out_channels, 
                              kernel_size, 
                              padding=self.padding, 
                              dilation=dilation, 
                              bias=False, 
                              **kwargs)
    
    def forward(self, input_):
        return self.conv(input_)[:, :, :-self.padding] if self.padding else self.conv(input_)


# gated activation units
class GatedActivationUnit(nn.Module):
    def __init__(self):
        super(GatedActivationUnit, self).__init__()
    
    def forward(self, input_):
        return torch.tanh(input_) * torch.sigmoid(input_)
