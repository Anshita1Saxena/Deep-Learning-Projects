from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import os


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, 
                 embed_dim=768):
        super().__init__()
        '''
        I am in PatchEmbed Initialization....
        img_size------> 32
        patch_size------> 16
        in_chans------> 3
        embed_dim------> 256
        '''
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Uncomment this line and replace ? with correct values
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch, num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC  [Batchsize, H*W, Channel]
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        '''
        I am in MLP Initialization....
        in_features------> 4
        hidden_features------> 128 ------>0.5*256(token dimension)
        act_layer------> <class 'torch.nn.modules.activation.GELU'>
        drop------> 0.0
        '''
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super().__init__()
        '''
        I am in MixerBlock Initialization....
        dim------> 256
        seq_len------> 4
        mlp_ratio------> (0.5, 4.0)
        activation------> gelu
        drop------> 0.0
        '''
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        # dim=256 and seq_len(num_of_blocks)=4
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)  # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x.shape = torch.Size([128, 64, 256])
        # hidden dimension/patch embedding = Number of columns of the table
        y = self.norm1(x)  # [batchsize, n_patches, hidden dimension/patch embedding]
        # For applying tokens, we need patches to be the last dimension
        y = y.permute(0, 2, 1)  # [batchsize, patch embedding, n_patches]
        y = self.mlp_tokens(y)  # [batchsize, patch embedding, n_patches]
        # Channels will be again at the last dimension
        y = y.permute(0, 2, 1)  # [batchsize, n_patches, patch embedding]
        # Take the result of the token mixing and add it to the residual (original)
        x = x + y  # [batchsize, n_patches, patch embedding]
        y = self.norm2(x)  # [batchsize, n_patches, patch embedding]
        # apply channel mixing and add it to the residual
        return x + self.mlp_channels(y)  # [batchsize, n_patches, patch embedding]


class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu'):
        super().__init__()
        '''
        I am in MLPMixer Initialization....
        num_classes------> 10
        img_size------> 32
        patch_size------> 16 (When I was looking at dimensions)
        num_blocks------> 4
        drop_rate------> 0.0
        activation------> gelu
        '''''
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """
        # step1: Go through the patch embedding
        # step 2 Go through the mixer blocks
        # step 3 go through layer norm
        # step 4 Global averaging spatially
        # Classification
        # Each pixel = Patch, channel dim represents the patch embedding
        # [batchsize, hidden_dim, n_patches ** (1/2), n_patches ** (1/2)]
        out = self.patchemb(images)  # [Batchsize, patch(width*height=token), hidden_dim(channel)]
        out = self.blocks(out)  # [Batchsize, patch, hidden_dim]
        out = self.norm(out)    # [Batchsize, patch, hidden_dim]
        # Average Pooling by average along the token dimension
        out = out.mean(dim=1)   # [Batchsize, hidden_dim]
        # Classification
        out = self.head(out)    # [Batchsize, hidden_dim]
        return out

    def visualize(self, logdir):
        """ Visualize the token mixer layer 
        in the desired directory """
        # Way to get the number of blocks
        # print(len(self.blocks))
        # Way to get the parameters of 1st block without names (Only Generator Object printed)
        # print('parameters------>', self.blocks[0].parameters())
        # Way to get the parameters of 1st block with names (Only Generator Object printed)
        # print('named_parameters----->', self.blocks[0].named_parameters('l'))
        # Way to get the parameters with values without names
        # print('parameters starts---->')
        # for idx, m in enumerate(self.blocks[0].parameters()):
        #     print(idx, '----->', m)
        # Way to get the parameters with values with names
        # print('mlp_tokens.fc1.weight---->')
        # print(self.blocks[0].get_parameter('mlp_tokens.fc1.weight'))
        # print(self.blocks[0].get_parameter('mlp_tokens.fc1.weight').data.shape)
        filter = self.blocks[0].get_parameter('mlp_tokens.fc1.weight').data
        number_of_patches = int(self.patchemb.img_size / self.patchemb.patch_size)
        filter = self.blocks[0].get_parameter('mlp_tokens.fc1.weight').data
        filter_reshape = filter.view(filter.shape[0], number_of_patches,
                                     number_of_patches)
        # Detach the tensor to avoid breaking the computation graph for numpy
        filter_reshape1 = filter_reshape.detach()
        filter_reshape1 = filter_reshape1.cpu()
        filter_reshape1 = filter_reshape1.data.numpy()
        # print(filters1.shape)
        # # normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filter_reshape1.min(), filter_reshape1.max()
        filter_reshape1 = (filter_reshape1 - f_min) / (f_max - f_min)
        # filter_mean = np.mean(filter_reshape1, axis=3)
        # print(filter_mean.shape)
        # plot first few filters
        n_filters, ix = 1, 16
        for i in range(ix):
            # get the filter
            for j in range(8):
                f = filter_reshape1[n_filters - 1, :, :]
                ax = plt.subplot(ix, 8, n_filters)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(f, cmap='gray')
                n_filters += 1
        # show the figure
        fig1 = plt.gcf()
        plt.rcParams["figure.figsize"] = (30, 50)
        plt.show()
        plt.draw()
        fig1.savefig(os.path.join(logdir, f'visualize_kernel.png'))
        # Way to get only module structure
        # print('named_module starts---->')
        # for idx, m in enumerate(self.blocks[0].named_modules()):
        #     print(idx, '----->', m)
        # for idx, m in enumerate(self.blocks[0].named_parameters()):
        #     if m.__getitem__(0) == 'mlp_tokens.fc1.weight':
        #         print(idx, '----->', m.__getitem__(1))
        #         print(type(m.__getitem__(1)))
        #         print(type(m.__getitem__(1).data))
        #         print(m.__getitem__(1).data.shape)
        #         print(m.__getitem__(1).data)

