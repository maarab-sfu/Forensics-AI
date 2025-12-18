import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
import utlz

def D_unet_arch(ch=64, attention='64',ksize='333333', dilation='111111',out_channel_multiplier=1):
    arch = {}

    n = 2

    ocm = out_channel_multiplier

    # covers bigger perceptual fields
    arch[128]= {'in_channels' :       [3] + [ch*item for item in       [1, 2, 4, 8, 16, 8*n, 4*2, 2*2, 1*2,1]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 8,   4,   2,    1,  1]],
                             'downsample' : [True]*5 + [False]*5,
                             'upsample':    [False]*5+ [True] *5,
                             'resolution' : [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,11)}}


    arch[256] = {'in_channels' :            [3] + [ch*item for item in [1, 2, 4, 8, 8, 16, 8*2, 8*2, 4*2, 2*2, 1*2  , 1         ]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 8,   8,   4,   2,   1,   1          ]],
                             'downsample' : [True] *6 + [False]*6 ,
                             'upsample':    [False]*6 + [True] *6,
                             'resolution' : [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256 ],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,13)}}



    return arch


class Unet_Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                             D_kernel_size=3, D_attn='64', n_classes=1000,
                             num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                             D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                             SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                             D_init='ortho', skip_init=False, D_param='SN', decoder_skip_connection = True, **kwargs):
        super(Unet_Discriminator, self).__init__()


        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16



        if self.resolution==128:
            self.save_features = [0,1,2,3,4]
        elif self.resolution==256:
            self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4
        # Architecture
        self.arch = D_unet_arch(self.ch, self.attention , out_channel_multiplier = self.out_channel_multiplier  )[resolution]

        self.unconditional = kwargs["unconditional"]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)

            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                            eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                             out_channels=self.arch['out_channels'][index],
                                             which_conv=self.which_conv,
                                             wide=self.D_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            elif self.arch["upsample"][index]:
                upsample_function = (functools.partial(F.interpolate, scale_factor=2, mode="nearest") #mode=nearest is default
                                    if self.arch['upsample'][index] else None)

                self.blocks += [[layers.GBlock2(in_channels=self.arch['in_channels'][index],
                                                         out_channels=self.arch['out_channels'][index],
                                                         which_conv=self.which_conv,
                                                         #which_bn=self.which_bn,
                                                         activation=self.activation,
                                                         upsample= upsample_function, skip_connection = True )]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition: #index < 5
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                print("index = ", index)
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                                                         self.which_conv)]


        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])


        last_layer = nn.Conv2d(self.ch*self.out_channel_multiplier,1,kernel_size=1)
        self.blocks.append(last_layer)
        #
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

        self.linear_middle = self.which_linear(16*self.ch, output_dim)
        # Embedding for projection discrimination
        #if not kwargs["agnostic_unet"] and not kwargs["unconditional"]:
        #    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1]+extra)
        if not kwargs["unconditional"]:
            self.embed_middle = self.which_embedding(self.n_classes, 16*self.ch)
            self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        ###
        print("_____params______")
        for name, param in self.named_parameters():
            print(name, param.size())

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)



    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index==6 :
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==7:
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==8:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[1]),dim=1)

            if self.resolution == 256:
                if index==7:
                    h = torch.cat((h,residual_features[5]),dim=1)
                elif index==8:
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==10:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==11:
                    h = torch.cat((h,residual_features[1]),dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index==self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(self.activation(h), [2, 3])
                # Get initial class-unconditional output
                bottleneck_out = self.linear_middle(h_)
                # Get projection of final featureset onto class vectors and add to evidence
                if self.unconditional:
                    projection = 0
                else:
                    # this is the bottleneck classifier c
                    emb_mid = self.embed_middle(y)
                    projection = torch.sum(emb_mid * h_, 1, keepdim=True)
                bottleneck_out = bottleneck_out + projection

        out = self.blocks[-1](h)

        if self.unconditional:
            proj = 0
        else:
            emb = self.embed(y)
            emb = emb.view(emb.size(0),emb.size(1),1,1).expand_as(h)
            proj = torch.sum(emb * h, 1, keepdim=True)
            ################
        out = out + proj

        out = out.view(out.size(0),1,self.resolution,self.resolution)

        return out, bottleneck_out



class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, rev):
        x = x[0]
        if not rev:
            B, C, H, W = x.shape
            x = x.reshape(B, C, H // 2, 2, W // 2, 2)  # factor spatial dim
            x = x.permute(0, 1, 3, 5, 2, 4)  # transpose to (B, C, 2, 2, H//2, W//2)
            x = x.reshape(B, 4 * C, H // 2, W // 2)  # aggregate spatial dim factors into channels
            return [x]
        else:
            B, C, H, W = x.shape
            x = x.reshape(B, C // 4, 2, 2, H, W)  # factor channel dim
            x = x.permute(0, 1, 4, 2, 5, 3)  # transpose to (B, C//4, H, 2, W, 2)
            x = x.reshape(B, C // 4, 2 * H, 2 * W)  # aggregate channel dim factors into spatial dims
            return [x]


class Unsqueeze(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, rev):
        x = x[0]
        if not rev:
            B, C, H, W = x.shape
            x = x.reshape(B, C // 4, 2, 2, H, W)  # factor channel dim
            x = x.permute(0, 1, 4, 2, 5, 3)  # transpose to (B, C//4, H, 2, W, 2)
            x = x.reshape(B, C // 4, 2 * H, 2 * W)  # aggregate channel dim factors into spatial dims
            return [x]
        else:
            B, C, H, W = x.shape
            x = x.reshape(B, C, H // 2, 2, W // 2, 2)  # factor spatial dim
            x = x.permute(0, 1, 3, 5, 2, 4)  # transpose to (B, C, 2, 2, H//2, W//2)
            x = x.reshape(B, 4 * C, H // 2, W // 2)  # aggregate spatial dim factors into channels
            return [x]


class Gaussianize(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.net = ResidualDenseBlock_out(n_channels, 4*n_channels)  # computes the parameters of Gaussian
        self.clamp = 1.
        self.affine_eps = 0.0001

    def forward(self, x1, x2, rev=False):
        if not rev:
            h = self.net(x1)
            m, s = h[:, 0::2, :, :], h[:, 1::2, :, :]          # split along channel dims
            z2 = (x2 - m) / self.e(s)                # center and scale; log prob is computed at the model forward
            return z2
        else:
            z2 = x2
            h = self.net(x1)
            m, s = h[:, 0::2, :, :], h[:, 1::2, :, :]
            x2 = m + z2 * self.e(s)
            return x2

    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps


class RNVPCouplingBlock(nn.Module):

    def __init__(self, dims_in, subnet_constructor=None, clamp=1.0):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.affine_eps = 0.0001
        # self.max_s = exp(clamp)
        # self.min_s = exp(-clamp)

        self.s1 = subnet_constructor(self.split_len1, self.split_len2)
        self.t1 = subnet_constructor(self.split_len1, self.split_len2)
        self.s2 = subnet_constructor(self.split_len2, self.split_len1)
        self.t2 = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            x2_c = x2
            s2, t2 = self.s2(x2_c), self.t2(x2_c)
            y1 = self.e(s2) * x1 + t2
            y1_c = y1
            s1, t1 = self.s1(y1_c), self.t1(y1_c)
            y2 = self.e(s1) * x2 + t1
            self.last_s = [s1, s2]
        else:
            x1_c = x1
            s1, t1 = self.s1(x1_c), self.t1(x1_c)
            y2 = (x2 - t1) / self.e(s1)
            y2_c = y2
            s2, t2 = self.s2(y2_c), self.t2(y2_c)
            y1 = (x1 - t2) / self.e(s2)
            self.last_s = [s1, s2]

        return [torch.cat((y1, y2), 1)]


class HaarDownsampling(nn.Module):

    def __init__(self, dims_in, order_by_wavelet=False, rebalance=1.):
        super().__init__()

        self.in_channels = dims_in[0][0]
        self.fac_fwd = 0.5 * rebalance
        self.fac_rev = 0.5 / rebalance
        self.haar_weights = torch.ones(4,1,2,2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

        self.permute = order_by_wavelet
        self.last_jac = None

        if self.permute:
            permutation = []
            for i in range(4):
                permutation += [i+4*j for j in range(self.in_channels)]

            self.perm = torch.LongTensor(permutation)
            self.perm_inv = torch.LongTensor(permutation)

            for i, p in enumerate(self.perm):
                self.perm_inv[p] = i

    def forward(self, x, rev=False):
        if not rev:
            # self.last_jac = self.elements / 4 * (np.log(16.) + 4 * np.log(self.fac_fwd))
            out = F.conv2d(x[0], self.haar_weights,
                           bias=None, stride=2, groups=self.in_channels)
            if self.permute:
                return [out[:, self.perm] * self.fac_fwd]
            else:
                return [out * self.fac_fwd]

        else:
            # self.last_jac = self.elements / 4 * (np.log(16.) + 4 * np.log(self.fac_rev))
            if self.permute:
                x_perm = x[0][:, self.perm_inv]
            else:
                x_perm = x[0]

            return [F.conv_transpose2d(x_perm * self.fac_rev, self.haar_weights,
                                     bias=None, stride=2, groups=self.in_channels)]

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return self.last_jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c*4, w//2, h//2
        self.elements = c*w*h
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]


class HaarUpsampling(nn.Module):

    def __init__(self, dims_in):
        super().__init__()

        self.in_channels = dims_in[0][0] // 4
        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights *= 0.5
        self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if rev:
            return [F.conv2d(x[0], self.haar_weights,
                             bias=None, stride=2, groups=self.in_channels)]
        else:
            return [F.conv_transpose2d(x[0], self.haar_weights,
                                       bias=None, stride=2,
                                       groups=self.in_channels)]

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c//4, w*2, h*2
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            utlz.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            utlz.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        utlz.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

class ResidualDenseBlock_out(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(channel_in, 32, 3, 1, 1, bias=bias))
        self.conv2 = spectral_norm(nn.Conv2d(channel_in + 32, 32, 3, 1, 1, bias=bias))
        self.conv3 = spectral_norm(nn.Conv2d(channel_in + 2 * 32, 32, 3, 1, 1, bias=bias))
        self.conv4 = nn.Conv2d(channel_in + 3 * 32, channel_out, 3, 1, 1, bias=bias)
        
        self.elu = nn.ELU(inplace=True)
        # initialization
        if init == 'xavier':
            utlz.initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            utlz.initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        utlz.initialize_weights(self.conv4, 0)

    def forward(self, x):
        x1 = self.elu(self.conv1(x))
        x2 = self.elu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.elu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4


class IINet(nn.Module):
    def __init__(self, dims_in=[[4, 64, 64]], down_num=3, block_num=[4, 4, 4]):
        super(IINet, self).__init__()

        operations = []

        current_dims = dims_in
        for i in range(down_num):
            b = HaarDownsampling(current_dims)
            # b = Squeeze()
            operations.append(b)
            current_dims[0][0] = current_dims[0][0] * 4
            current_dims[0][1] = current_dims[0][1] // 2
            current_dims[0][2] = current_dims[0][2] // 2
            for j in range(block_num[i]):
                b = RNVPCouplingBlock(current_dims, subnet_constructor=ResidualDenseBlock_out, clamp=1.0)
                operations.append(b)
        block_num = block_num[:-1][::-1]
        block_num.append(0)
        for i in range(down_num):
            b = HaarUpsampling(current_dims)
            # b = Unsqueeze()
            operations.append(b)
            current_dims[0][0] = current_dims[0][0] // 4
            current_dims[0][1] = current_dims[0][1] * 2
            current_dims[0][2] = current_dims[0][2] * 2
            for j in range(block_num[i]):
                b = RNVPCouplingBlock(current_dims, subnet_constructor=DenseBlock, clamp=1.0)
                operations.append(b)

        self.operations = nn.ModuleList(operations)
        # self.guassianize = Gaussianize(1)

    def forward(self, x, rev=False):
        out = x
        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
            # g, z = out[0][:, [0], :, :], out[0][:, 1:, :, :]
            # z = self.guassianize(x1=g, x2=z, rev=rev)
            # out = [torch.cat((g, z), dim=1)]
        else:
            # g, z = out[0][:, [0], :, :], out[0][:, 1:, :, :]
            # z = self.guassianize(x1=g, x2=z, rev=rev)
            # out = [torch.cat((g, z), dim=1)]
            for op in reversed(self.operations):
                out = op.forward(out, rev)
        return out

# Define a Convolutional Block with Spectral Normalization and ELU activation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        return x

class ForgeryDetector(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(ForgeryDetector, self).__init__()

        # Encoder
        self.haar_down1 = HaarDownsampling([[in_channels]])
        self.conv_block1 = ConvBlock(in_channels*4, 32)
        self.haar_down2 = HaarDownsampling([[32]])
        self.conv_block2 = ConvBlock(32*4, 32*4)
        self.haar_down3 = HaarDownsampling([[32*4]])
        self.conv_block3 = ConvBlock(32*4*4, 32*4*4)
        self.conv_block4 = ConvBlock(32*4*4, 32*4*4)

        # Decoder
        self.conv_block5 = ConvBlock(32*4*4, 32*4*4)
        self.conv_block6 = ConvBlock(32*4*4 + 32*4*4, 32*4*4)
        self.haar_up1 = HaarUpsampling([[32*4*4]])
        self.conv_block7 = ConvBlock(32*4 + 32*4, 32*4)
        self.haar_up2 = HaarUpsampling([[32*4]])
        self.conv_block8 = ConvBlock(32 + 32, 4 * in_channels)
        self.haar_up3 = HaarUpsampling([[4 * in_channels]])
        
        # Final Convolution
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        # print("x: ", x.shape)
        x1 = self.haar_down1([x])
        # print("x1: ", x1[0].shape)
        x2 = self.conv_block1(x1[0])

        x3 = self.haar_down2([x2])
        x4 = self.conv_block2(x3[0])

        x5 = self.haar_down3([x4])
        x6 = self.conv_block3(x5[0])
        x7 = self.conv_block4(x6)
        x8 = self.conv_block5(x7)
        
        x9 = self.conv_block6(torch.cat((x8,x6), dim = 1))
        x10 = self.haar_up1([x9])

        x11 = self.conv_block7(torch.cat((x10[0],x4), dim = 1))
        x12 = self.haar_up2([x11])

        x13 = self.conv_block8(torch.cat((x12[0],x2), dim = 1))
        x14 = self.haar_up3([x13])

        out = self.final_conv(x14[0])

        return out


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    img_name = "/media/ruizhao/programs/datasets/Denoising/testset/Kodak24/kodim04.png"

    image_c = cv2.imread(img_name, cv2.IMREAD_COLOR)[:, :, ::-1] / 255
    image_c = cv2.resize(image_c, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
    tensor_c = torch.from_numpy(image_c.transpose((2, 0, 1)).astype(np.float32))

    batch_c = tensor_c.unsqueeze(0)

    net = Inveritible_Decolorization()
    net.eval()
    net = net.cuda()

    with torch.no_grad():
        batch_x = net(x=[batch_c.cuda()], rev=False)
        batch_y = net(x=batch_x, rev=True)[0]

    print((batch_y.cpu() - batch_c).sum())

    plt.figure(0)
    plt.imshow(image_c)
    plt.show()
    r = batch_y.cpu().numpy()[0, :, :, :]
    plt.figure(1)
    plt.imshow(r.transpose((1, 2, 0)))
    plt.show()
    print("done")
