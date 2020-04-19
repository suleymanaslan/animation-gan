import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator2D(nn.Module):
    def __init__(self, latent_size=100, input_channels=108, feature_map_size=64):
        super(Generator2D, self).__init__()
        net_channels = [latent_size,
                        feature_map_size*8, 
                        feature_map_size*4, 
                        feature_map_size*2, 
                        feature_map_size*1,
                        input_channels]
#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(net_channels[0], net_channels[1], 4, 1, 0, bias=False), nn.BatchNorm2d(net_channels[1]), nn.ReLU(True),
#             nn.ConvTranspose2d(net_channels[1], net_channels[2], 4, 2, 1, bias=False), nn.BatchNorm2d(net_channels[2]), nn.ReLU(True),
#             nn.ConvTranspose2d(net_channels[2], net_channels[3], 4, 2, 1, bias=False), nn.BatchNorm2d(net_channels[3]), nn.ReLU(True),
#             nn.ConvTranspose2d(net_channels[3], net_channels[4], 4, 2, 1, bias=False), nn.BatchNorm2d(net_channels[4]), nn.ReLU(True),
#             nn.ConvTranspose2d(net_channels[4], net_channels[5], 4, 2, 1, bias=False), nn.Tanh()
#         )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(net_channels[0], net_channels[1], 4, 1, 0), nn.BatchNorm2d(net_channels[1]), nn.ReLU(True),
            nn.ConvTranspose2d(net_channels[1], net_channels[2], 4, 2, 1), nn.BatchNorm2d(net_channels[2]), nn.ReLU(True),
            nn.ConvTranspose2d(net_channels[2], net_channels[3], 4, 2, 1), nn.BatchNorm2d(net_channels[3]), nn.ReLU(True),
            nn.ConvTranspose2d(net_channels[3], net_channels[4], 4, 2, 1), nn.BatchNorm2d(net_channels[4]), nn.ReLU(True),
            nn.ConvTranspose2d(net_channels[4], net_channels[5], 4, 2, 1), nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Generator3D(nn.Module):
    def __init__(self, latent_size=100, input_channels=3, feature_map_size=64):
        super(Generator3D, self).__init__()
        net_channels = [latent_size,
                        feature_map_size*8, 
                        feature_map_size*4, 
                        feature_map_size*2, 
                        feature_map_size*1,
                        input_channels]
        self.main = nn.Sequential(
            nn.ConvTranspose3d(net_channels[0], net_channels[1], (2, 4, 4), 1, 0, bias=False), nn.BatchNorm3d(net_channels[1]), nn.ReLU(True),
            nn.ConvTranspose3d(net_channels[1], net_channels[2], (2, 4, 4), 2, (0, 1, 1), bias=False), nn.BatchNorm3d(net_channels[2]), nn.ReLU(True),
            nn.ConvTranspose3d(net_channels[2], net_channels[3], (2, 4, 4), 2, (0, 1, 1), bias=False, output_padding=(1, 0, 0)), nn.BatchNorm3d(net_channels[3]), nn.ReLU(True),
            nn.ConvTranspose3d(net_channels[3], net_channels[4], (2, 4, 4), 2, (0, 1, 1), bias=False), nn.BatchNorm3d(net_channels[4]), nn.ReLU(True),
            nn.ConvTranspose3d(net_channels[4], net_channels[5], (2, 4, 4), 2, (0, 1, 1), bias=False), nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator2D(nn.Module):
    def __init__(self, input_channels=108, output_channels=1, feature_map_size=64, groups=1):
        super(Discriminator2D, self).__init__()
        net_channels = [input_channels,
                        feature_map_size*1, 
                        feature_map_size*2, 
                        feature_map_size*4, 
                        feature_map_size*8,
                        output_channels]
        if groups == 1:
            assert output_channels == 1
#             self.main = nn.Sequential(
#                 nn.Conv2d(net_channels[0], net_channels[1], 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(net_channels[1], net_channels[2], 4, 2, 1, bias=False), nn.BatchNorm2d(net_channels[2]), nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(net_channels[2], net_channels[3], 4, 2, 1, bias=False), nn.BatchNorm2d(net_channels[3]), nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(net_channels[3], net_channels[4], 4, 2, 1, bias=False), nn.BatchNorm2d(net_channels[4]), nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(net_channels[4], net_channels[5], 4, 1, 0, bias=False), nn.Sigmoid()
#             )
            self.main = nn.Sequential(
                nn.Conv2d(net_channels[0], net_channels[1], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(net_channels[1], net_channels[2], 4, 2, 1), nn.BatchNorm2d(net_channels[2]), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(net_channels[2], net_channels[3], 4, 2, 1), nn.BatchNorm2d(net_channels[3]), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(net_channels[3], net_channels[4], 4, 2, 1), nn.BatchNorm2d(net_channels[4]), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(net_channels[4], net_channels[5], 4, 1, 0), nn.Sigmoid()
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(net_channels[0], net_channels[1], 4, 2, 1, groups=groups, bias=False), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(net_channels[1], net_channels[2], 4, 2, 1, groups=groups, bias=False), nn.BatchNorm2d(net_channels[2]), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(net_channels[2], net_channels[3], 4, 2, 1, groups=groups, bias=False), nn.BatchNorm2d(net_channels[3]), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(net_channels[3], net_channels[4], 4, 2, 1, groups=groups, bias=False), nn.BatchNorm2d(net_channels[4]), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(net_channels[4], net_channels[5], 4, 1, 0, groups=groups, bias=False), nn.BatchNorm2d(net_channels[5]), nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(), nn.Linear(net_channels[5], 1), nn.Sigmoid()
            )
            

    def forward(self, input):
        return self.main(input)


class DiscriminatorTemporal(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, feature_map_size=32, batch_size=8, sequence_length=36):
        super(DiscriminatorTemporal, self).__init__()
        
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.feature_map_size = feature_map_size
        
        net_channels = [input_channels,
                        feature_map_size*1, 
                        feature_map_size*2, 
                        feature_map_size*4, 
                        feature_map_size*8,
                        feature_map_size*16,
                        output_channels]

        self.main = nn.Sequential(
            nn.Conv2d(net_channels[0], net_channels[1], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[1], net_channels[2], 4, 2, 1), nn.BatchNorm2d(net_channels[2]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[2], net_channels[3], 4, 2, 1), nn.BatchNorm2d(net_channels[3]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[3], net_channels[4], 4, 2, 1), nn.BatchNorm2d(net_channels[4]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[4], net_channels[5], 4, 1, 0), nn.BatchNorm2d(net_channels[5]), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        
        self.temporal = nn.Sequential(
            nn.Conv1d(net_channels[5], net_channels[4], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(net_channels[4], net_channels[3], 4, 2, 1), nn.BatchNorm1d(net_channels[3]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(net_channels[3], net_channels[2], 4, 2, 1), nn.BatchNorm1d(net_channels[2]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(net_channels[2], net_channels[6], 4, 1, 0), nn.Sigmoid()
        )
    
    def forward(self, input):
        conv_out = self.main(input)
        conv_out = conv_out.reshape(self.batch_size, self.sequence_length, self.feature_map_size*16).permute(0, 2, 1)
        temporal_out = self.temporal(conv_out)
        return temporal_out


class DiscriminatorSheet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, feature_map_size=64):
        super(DiscriminatorSheet, self).__init__()
        net_channels = [input_channels,
                        feature_map_size*1, 
                        feature_map_size*2, 
                        feature_map_size*4, 
                        feature_map_size*8, 
                        feature_map_size*16,
                        feature_map_size*32,
                        output_channels]
        self.main = nn.Sequential(
            nn.Conv2d(net_channels[0], net_channels[1], 5, (2, 3), 0, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[1], net_channels[2], 5, (2, 3), 0, bias=False), nn.BatchNorm2d(net_channels[2]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[2], net_channels[3], 5, 2, 0, bias=False), nn.BatchNorm2d(net_channels[3]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[3], net_channels[4], 5, 2, 0, bias=False), nn.BatchNorm2d(net_channels[4]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[4], net_channels[5], 3, 2, 0, bias=False), nn.BatchNorm2d(net_channels[5]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[5], net_channels[6], 3, 2, 0, bias=False), nn.BatchNorm2d(net_channels[6]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[6], net_channels[7], 2, 1, 0, bias=False), nn.Sigmoid()
        )
            
    def forward(self, input):
        return self.main(input)


class Discriminator3D(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, feature_map_size=64):
        super(Discriminator3D, self).__init__()
        net_channels = [input_channels,
                        feature_map_size*1, 
                        feature_map_size*2, 
                        feature_map_size*4, 
                        feature_map_size*8,
                        output_channels]
        self.main = nn.Sequential(
            nn.Conv3d(net_channels[0], net_channels[1], (2, 4, 4), 2, (0, 1, 1), bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(net_channels[1], net_channels[2], (2, 4, 4), 2, (0, 1, 1), bias=False), nn.BatchNorm3d(net_channels[2]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(net_channels[2], net_channels[3], (2, 4, 4), 2, (0, 1, 1), bias=False), nn.BatchNorm3d(net_channels[3]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(net_channels[3], net_channels[4], (2, 4, 4), 2, (0, 1, 1), bias=False), nn.BatchNorm3d(net_channels[4]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(net_channels[4], net_channels[5], (2, 4, 4), 1, 0, bias=False), nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

