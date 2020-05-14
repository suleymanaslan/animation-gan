# adapted from https://github.com/facebookresearch/pytorch_GAN_zoo

import torch
import torch.nn as nn
from pytorch_GAN_zoo.models.networks.custom_layers import EqualizedConv2d, EqualizedLinear, NormalizationLayer, Upscale2d
from pytorch_GAN_zoo.models.utils.utils import num_flat_features


class PGANGenerator(nn.Module):
    def __init__(self):
        super(PGANGenerator, self).__init__()
        self.depth_scale0 = 512
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.dim_output = 3
        self.dim_latent = 512
        self.scales_depth = [self.depth_scale0]
        
        self.scale_layers = nn.ModuleList()
        
        self.to_rgb_layers = nn.ModuleList()
        self.to_rgb_layers.append(EqualizedConv2d(self.depth_scale0, self.dim_output, 1, equalized=self.equalized_lr, initBiasToZero=self.init_bias_to_zero))
        
        self.format_layer = EqualizedLinear(self.dim_latent, 16 * self.scales_depth[0], equalized=self.equalized_lr, initBiasToZero=self.init_bias_to_zero)
        
        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(EqualizedConv2d(self.depth_scale0, self.depth_scale0, 3, equalized=self.equalized_lr, initBiasToZero=self.init_bias_to_zero, padding=1))
        
        self.alpha = 0
        
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        
        self.normalization_layer = NormalizationLayer()
        
        self.generation_activation = None
        
    def forward(self, x):
        print(f"input, x:{x.shape}")
        
        x = self.normalization_layer(x)
        x = x.view(-1, num_flat_features(x))
        x = self.leaky_relu(self.format_layer(x))
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalization_layer(x)
        print(f"format, x:{x.shape}")
        
        for conv_layer in self.group_scale0:
            x = self.leaky_relu(conv_layer(x))
            print(f"conv, x:{x.shape}")
            x = self.normalization_layer(x)
        
        if self.alpha > 0 and len(self.scale_layers) == 1:
            y = self.to_rgb_layers[-2](x)
            print(f"to rgb, y:{y.shape}")
            y = Upscale2d(y)
            print(f"upscale, y:{y.shape}")
            
        for scale, layer_group in enumerate(self.scale_layers, 0):
            print(f"scale:{scale}, layer_group:{layer_group}")
            x = Upscale2d(x)
            print(f"upscale, x:{x.shape}")
            for conv_layer in layer_group:
                x = self.leaky_relu(conv_layer(x))
                print(f"conv, x:{x.shape}")
                x = self.normalization_layer(x)
            if self.alpha > 0 and scale == (len(self.scale_layers) - 2):
                y = self.to_rgb_layers[-2](x)
                print(f"to rgb, y:{y.shape}")
                y = Upscale2d(y)
                print(f"upscale, y:{y.shape}")
                
        x = self.to_rgb_layers[-1](x)
        print(f"to rgb, x:{x.shape}")
        
        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x
            print(f"blend, x:{x.shape}")
            
        if self.generation_activation is not None:
            x = self.generation_activation(x)
            print(f"g act, x:{x.shape}")
        
        return x
