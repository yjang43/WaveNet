import torch
import torch.nn as nn
from module import CausalConv1d, GatedActivationUnit


class WaveNet(nn.Module):
    def __init__(
        self,
        num_block=4,
        num_layer=10,
        class_dim=256,
        residual_dim=32,
        dilation_dim=32,
        skip_dim=256,
        kernel_size=2,
        bias=False
    ):
        super(WaveNet, self).__init__()
        self.start_conv = nn.Conv1d(in_channels=class_dim, 
                                    out_channels=residual_dim, 
                                    kernel_size=1, 
                                    bias=bias)
        
        self.stack = nn.ModuleList()
        for b in range(num_block):
            dilation = 1
            for k in range(num_layer):
                
                layer = nn.Sequential(
                    CausalConv1d(in_channels=residual_dim,
                                 out_channels=dilation_dim, 
                                 kernel_size=kernel_size,
                                 dilation=dilation),
                    GatedActivationUnit(),
                    nn.Conv1d(in_channels=dilation_dim, 
                              out_channels=residual_dim, 
                              kernel_size=1, 
                              bias=bias)
                )
                
                self.stack.append(layer)
                dilation *= 2
        
        self.end_conv = nn.Sequential(
            nn.ReLU(), 
            nn.Conv1d(in_channels=residual_dim, 
                      out_channels=skip_dim, 
                      kernel_size=1, 
                      bias=bias),
            nn.ReLU(),
            nn.Conv1d(in_channels=skip_dim, 
                      out_channels=class_dim, 
                      kernel_size=1, 
                      bias=bias)
        )
        
        
    def forward(self, input_):
        residual = self.start_conv(input_)
        skips = torch.zeros_like(residual)
        
        for layer in self.stack:
            skip = layer(residual)
            residual = residual + skip
            skips = skips + skip
        logit = self.end_conv(skips)
        return logit

    
        