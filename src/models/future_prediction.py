import torch
import torch.nn as nn

from src.config import get_cfg
from src.layers.convolutions import Bottleneck, Block, DeepLabHead
from src.layers.temporal import SpatialGRU, Dual_GRU, BiGRU

class FuturePrediction(nn.Module):
    def __init__(self, in_channels, latent_dim, n_future, mixture=True, n_gru_blocks=2, n_res_layers=1):
        super(FuturePrediction, self).__init__()
        self.n_spatial_gru = n_gru_blocks
        self.cfg = get_cfg()

        gru_in_channels = latent_dim
        self.dual_grus = Dual_GRU(gru_in_channels, in_channels, n_future=n_future, mixture=mixture)
        self.res_blocks1 = nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)])

        self.spatial_grus = []
        self.res_blocks = []
        for i in range(self.n_spatial_gru):
            self.spatial_grus.append(SpatialGRU(in_channels, in_channels))
            if i < self.n_spatial_gru - 1:
                self.res_blocks.append(nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)]))
            else:
                self.res_blocks.append(DeepLabHead(in_channels, in_channels, 128))

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)


    def forward(self, x, state):
        """
        x: B, 1, 32, 200, 200
        state: B, P, 64, 200, 200
        """
        # x has shape (b, 1, c, h, w), state: torch.Tensor [b, n_present, hidden_size, h, w]
        if self.cfg.DEBUG: breakpoint()
        """
        dual_grus -> takes past input (P,c,h,w) and transforms to future (F,c,h,w)
        """
        x = self.dual_grus(x, state) # B, F, 64, 200, 200
        if self.cfg.DEBUG: breakpoint() 

        b, n_future, c, h, w = x.shape
        # x = self.res_blocks1(x.view(b * n_future, c, h, w)) 
        # B * F, 64, 200, 200
        if self.cfg.DEBUG: breakpoint()        
        x = x.view(b, n_future, c, h, w)

        # x: B, T, 64, 200, 200 -> all states represented
        x = torch.cat([state, x], dim=1)
        if self.cfg.DEBUG: breakpoint()        

        hidden_state = x[:, 0]
        # n_spatial_gru = 2
        for i in range(self.n_spatial_gru - 1):
            x = self.spatial_grus[i](x, hidden_state)
            if self.cfg.DEBUG: breakpoint()            
            b, s, c, h, w = x.shape
            # x = self.res_blocks[i](x.view(b*s, c, h, w))
            if self.cfg.DEBUG: breakpoint()                        
            x = x.view(b, s, c, h, w)
        if self.cfg.DEBUG: breakpoint()            
        return x