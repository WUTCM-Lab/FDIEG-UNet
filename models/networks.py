import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
import functools
from einops import rearrange, repeat
import models
from models.help_funcs import Transformer, TransformerDecoder, TransformerDecoderSEQ, mlp_head, SpaceCenter_Concentrate_Attention
import models.frft_gpu as frft


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


# 主函数
def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[], input_size=198):
    if args.net_G == 'unet_transformer':
        net = UNet(args=args, input_size=input_size)

    elif args.net_G == 'unet_transformer_u':
        net = UNet_u(args=args, input_size=input_size)

    elif args.net_G == 'unet_transformer_s':
        net = UNet_s(args=args, input_size=input_size)
    elif args.net_G == 'unet_transformer_f':
        net = UNet_f(args=args, input_size=input_size)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################
# u-net网络解码器定义（特征融合及transform解码器）
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1)
        # up-conv 2*2
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True)
        )

    def forward(self, high, low):
        x1 = self.up(high)
        offset = x1.size()[2] - low.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        # 计算应该填充多少（这里可以是负数）
        x2 = F.pad(low, padding)  # 这里相当于对低级特征做一个crop操作
        x1 = torch.cat((x1, x2), dim=1)  # 拼起来
        x1 = self.conv_relu(x1)  # 卷积
        return x1


class LayerCrossAttention(nn.Module):
    def __init__(self):
        super(LayerCrossAttention, self).__init__()
        self.fc_q = nn.Sequential(
            nn.Linear(9, 8, False),
            nn.ReLU(),
            nn.Linear(8, 16, False),
            # nn.Sigmoid()
        )
        self.fc_k = nn.Sequential(
            nn.Linear(16, 8, False),
            nn.ReLU(),
            nn.Linear(8, 16, False),
            # nn.Sigmoid()
        )
        self.fc_v = nn.Sequential(
            nn.Linear(16, 8, False),
            nn.ReLU(),
            nn.Linear(8, 16, False),
            # nn.Sigmoid()
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        q = self.fc_q(q)
        k = self.fc_k(k.squeeze(-1).squeeze(-1))
        v = self.fc_v(v.squeeze(-1).squeeze(-1))
        qk = self.softmax(q*k)
        qkv = qk*v
        return qkv


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, input_size, m):
        super(DeepInfoMaxLoss, self).__init__()
        self.a0 = nn.Conv2d(input_size * 2, input_size, kernel_size=1)
        self.a2 = nn.Conv2d(input_size, input_size, kernel_size=1)
        self.a1 = nn.Conv2d(input_size, 1, kernel_size=1)
        # self.a0 = nn.Conv2d(input_size, input_size//2, kernel_size=1)
        # self.a2 = nn.Conv2d(input_size//2, input_size//2, kernel_size=1)
        # self.a1 = nn.Conv2d(input_size//2, 1, kernel_size=1)
        # self.l0 = nn.Linear(input_size*m*m+input_size, input_size)
        # self.l1 = nn.Linear(input_size, input_size)
        # self.l2 = nn.Linear(input_size, 1)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = 1.0
        self.beta = 0.5

    def forward(self, x, y, f, r):
        # lay = self.avg_pool(x)
        # lay = lay.view(lay.shape[0], -1)
        # amp = y.view(y.shape[0], -1)
        # h = torch.cat((lay, amp), dim=1)
        # h = F.relu(self.l0(h))
        # h = F.relu(self.l1(h))
        # i = torch.cat((x, y), dim=1)
        # i = x + y
        i = y
        f = torch.cat((i, f), dim=1)
        r = torch.cat((i, r), dim=1)
        f = F.relu(self.a0(f))
        f = F.relu(self.a2(f))
        r = F.relu(self.a0(r))
        r = F.relu(self.a2(r))
        return ((F.softplus(-self.a1(r)).mean()) * 1.0 + (F.softplus(self.a1(f)).mean()) * 0.1) # * self.alpha
        # return F.softplus(-self.l2(h)).mean() * self.beta + F.softplus(-self.a1(i)).mean() *s


class SimilarityLearn(nn.Module):
    def __init__(self):
        super(SimilarityLearn, self).__init__()
        self.fc4 = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(6, 9),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(15, 25),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(28, 49),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.fcl4 = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.fcl3 = nn.Sequential(
            nn.Linear(6, 9),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.fcl2 = nn.Sequential(
            nn.Linear(15, 25),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.fcl1 = nn.Sequential(
            nn.Linear(28, 49),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.amp_layer1 = nn.Sequential(  # size->128
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.amp_layer2 = nn.Sequential(  # size->64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.amp_layer3 = nn.Sequential(  # size->32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.amp_layer4 = nn.Sequential(  # size->16
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def _forward_reshape_feature(self, x):
        b, l, c = x.shape
        h = int(math.sqrt(c))
        feature = rearrange(x, 'b c (d a) -> b c d a', a=h)
        return feature

    def forward(self, lay1, lay2, lay3, lay4, amp1, amp2, amp3, amp4):
        amp1 = self.amp_layer1(self._forward_reshape_feature(self.fc1(amp1)))
        amp2 = self.amp_layer2(self._forward_reshape_feature(self.fc2(amp2)))
        amp3 = self.amp_layer3(self._forward_reshape_feature(self.fc3(amp3)))
        amp4 = self.amp_layer4(self.fc4(amp4))
        lay1 = self._forward_reshape_feature(self.fcl1(lay1))
        lay2 = self._forward_reshape_feature(self.fcl2(lay2))
        lay3 = self._forward_reshape_feature(self.fcl3(lay3))
        lay4 = self.fcl4(lay4)
        # lay1 = self.layer1(lay1)
        # lay2 = self.layer2(lay2)
        # lay3 = self.layer3(lay3)
        # lay4 = self.layer4(lay4)
        return lay1, lay2, lay3, lay4, amp1, amp2, amp3, amp4


class SequentialTransformer(nn.Module):
    def __init__(self, patch_size, channels, enc_depth, dim_head):
        super(SequentialTransformer, self).__init__()
        self.s_pos_embedding = nn.Parameter(torch.randn(1, channels, patch_size * patch_size))
        mlp_dim = 64 * 2
        self.sequential_transformer = Transformer(dim=patch_size * patch_size, depth=enc_depth, heads=8,
                                                  dim_head=dim_head,
                                                  mlp_dim=mlp_dim, dropout=0)

    def forward(self, x):
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = x + self.s_pos_embedding
        x = self.sequential_transformer(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=h)
        return x


# encoder顺承光谱信息，并分离信息（第一次）（2d与1d同时处理）
class FftModel(nn.Module):
    def __init__(self, args, dim, shape, learning, mode):
        super(FftModel, self).__init__()
        (c, h, w) = shape
        self.dim = dim
        self.learning = learning
        self.sigmoid = nn.Sigmoid()
        self.mode = mode
        self.norm5 = nn.LayerNorm([c, h, w, 2])
        self.norm6 = nn.LayerNorm([9, 2])
        if self.learning:
            if self.dim == 1:
                self.complex_weight1d = nn.Parameter(torch.randn(1, 9, 2, dtype=torch.float32) * 0.02)
                self.norm1 = nn.LayerNorm(16)
            elif self.dim == 2:
                self.complex_weight2d = nn.Parameter(torch.randn(1, c, h, w, 2, dtype=torch.float32) * 0.02)
                self.norm2 = nn.LayerNorm([c, h, h])

    def extract_ampl_phase(self, fft_im):
        # fft_im: size should be bxcxhxwx2
        if self.dim == 1:
            fft_amp = fft_im[:, :, 0] ** 2 + fft_im[:, :, 1] ** 2
            fft_amp = torch.sqrt(fft_amp)
            fft_pha = torch.atan2(fft_im[:, :, 1], fft_im[:, :, 0])
        elif self.dim == 2:
            fft_amp = fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1] ** 2
            fft_amp = torch.sqrt(fft_amp)
            fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])

        return fft_amp, fft_pha

    def forward(self, x):
        if self.learning:
            if self.dim == 1:
                x = x.squeeze(-1).squeeze(-1)
                # fft_src = torch.fft.rfft(x, norm='ortho')
                # fft_src = torch.stack((fft_src.real, fft_src.imag), -1)
                # amp_src, pha_src = self.extract_ampl_phase(fft_src)

                x = torch.fft.rfft(x, dim=-1, norm='ortho')
                weight1d = torch.view_as_complex(self.complex_weight1d)
                x = x * weight1d

                fft_src = self.norm6(torch.stack((x.real, x.imag), -1))
                amp_src, pha_src = self.extract_ampl_phase(fft_src)

                x = torch.fft.irfft(x, dim=-1, norm='ortho')
                x = self.norm1(x)
                x = x.unsqueeze(-1).unsqueeze(-1)
            elif self.dim == 2:
                # if self.mode == 'space':
                #     fft_src = torch.fft.rfft2(x, norm='ortho')
                #     fft_src = torch.stack((fft_src.real, fft_src.imag), -1)
                #     amp_src, pha_src = self.extract_ampl_phase(fft_src)
                # elif self.mode == 'seq':
                #     fft_src = torch.fft.rfft(x, dim=1, norm='ortho')
                #     fft_src = torch.stack((fft_src.real, fft_src.imag), -1)
                #     amp_src, pha_src = self.extract_ampl_phase(fft_src)
                x = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
                weight2d = torch.view_as_complex(self.complex_weight2d)
                x = x * weight2d

                if self.mode == 'space':
                    fft_src = self.norm5(torch.stack((x.real, x.imag), -1))
                    amp_src, pha_src = self.extract_ampl_phase(fft_src)

                x = torch.fft.irfftn(x, s=(32, 3, 3), dim=(1, 2, 3), norm='ortho')
                x = self.norm2(x)
        else:
            if self.dim == 2:
                if self.mode == 'space':
                    fft_src = torch.fft.rfftn(x.clone(), dim=(1,2,3), norm='ortho')
                    fft_src = torch.stack((fft_src.real, fft_src.imag), -1)
                    amp_src, pha_src = self.extract_ampl_phase(fft_src.clone())
                elif self.mode == 'seq':
                    fft_src = torch.fft.rfft(x, dim=1, norm='ortho')
                    fft_src = torch.stack((fft_src.real, fft_src.imag), -1)
                    amp_src, pha_src = self.extract_ampl_phase(fft_src)
            elif self.dim == 1:
                fft_src = torch.fft.rfft(x.clone(), norm='ortho')
                fft_src = torch.stack((fft_src.real, fft_src.imag), -1)
                amp_src, pha_src = self.extract_ampl_phase(fft_src.clone())
        return x, amp_src, pha_src


class UNet(nn.Module):
    def __init__(self, args, input_size, enc_depth=1, dim_head=64):
        super().__init__()

        self.dim_head = dim_head
        self.enc_depth = enc_depth
        dim = 64
        mlp_dim = dim * 2
        self.pool_mode = None

        self.layer1 = nn.Sequential(  # 9->7 size->128
            nn.Conv2d(input_size, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(  # 7->5 128->64
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(  # 5->3 64->32
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(  # 3->1 32->16
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # self.res1 = nn.Sequential(  # 9->7 size->128
        #     nn.Conv2d(256, 128, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        # self.res2 = nn.Sequential(  # 7->5 128->64
        #     nn.Conv2d(128, 64, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # self.res3 = nn.Sequential(  # 5->3 64->32
        #     nn.Conv2d(64, 32, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        # )
        # self.res4 = nn.Sequential(  # 3->1 32->16
        #     nn.Conv2d(32, 16, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        # )

        self.transformer1 = SequentialTransformer(patch_size=args.patches-2, channels=128,
                                                   enc_depth=enc_depth, dim_head=dim_head)
        self.transformer2 = SequentialTransformer(patch_size=args.patches-4, channels=64,
                                                   enc_depth=enc_depth, dim_head=dim_head)
        # self.transformer3 = Sequential_Transformer(patch_size=args.patches-6, channels=32,
        #                                            enc_depth=enc_depth, dim_head=dim_head)
        self.transformer_decoder1 = TransformerDecoder(dim=128, depth=1, heads=8, dim_head=64, mlp_dim=mlp_dim,
                                                      dropout=0, softmax=True, sub_pha=28, lent=49)
        self.transformer_decoder2 = TransformerDecoder(dim=64, depth=1, heads=8, dim_head=64, mlp_dim=mlp_dim,
                                                      dropout=0, softmax=True, sub_pha=15, lent=25)
        self.transformer_decoder3 = TransformerDecoder(dim=32, depth=1, heads=8, dim_head=64, mlp_dim=mlp_dim,
                                                      dropout=0, softmax=True, sub_pha=6, lent=9)
        # self.transformer_decoder1_seq = TransformerDecoderSEQ(dim=128, depth=1, heads=8, dim_head=64, mlp_dim=mlp_dim,
        #                                                dropout=0, softmax=True, sub_pha=65)
        # self.transformer_decoder2_seq = TransformerDecoderSEQ(dim=64, depth=1, heads=8, dim_head=64, mlp_dim=mlp_dim,
        #                                                dropout=0, softmax=True, sub_pha=33)
        # self.transformer_decoder3_seq = TransformerDecoderSEQ(dim=9, depth=1, heads=8, dim_head=64, mlp_dim=mlp_dim,
        #                                                dropout=0, softmax=True, sub_pha=6)
        self.transformer_decoder4 = LayerCrossAttention()
        self.fft_model_layer4 = FftModel(args=args, dim=1, shape=(1, 9, 2), learning=True, mode=None)
        self.fft_model_layer3 = FftModel(args=args, dim=2, shape=(32, 3, 2), learning=True, mode='space')
        self.fft_model_layer2 = FftModel(args=args, dim=2, shape=(64, 5, 3), learning=False, mode='space')
        self.fft_model_layer1 = FftModel(args=args, dim=2, shape=(128, 7, 4), learning=False, mode='space')
        self.new_loss_learning = SimilarityLearn()
        self.deepinfo128 = DeepInfoMaxLoss(input_size=128, m=7)
        self.deepinfo64 = DeepInfoMaxLoss(input_size=64, m=5)
        self.deepinfo32 = DeepInfoMaxLoss(input_size=32, m=3)
        self.deepinfo16 = DeepInfoMaxLoss(input_size=16, m=1)

        self.decoder4 = Decoder(16, 32)
        self.decoder3 = Decoder(64, 64)
        self.decoder2 = Decoder(128, 128)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = mlp_head(dim=256)  # 分类器
        self.classifier_pha = mlp_head(dim=16)
        self.sigmoid = nn.Sigmoid()

    def _forward_reshape_feature(self, x, mode):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [3, 3])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [3, 3])
        else:
            x = x
        if mode == 'space':
            b, l, c = x.shape
            h = int(math.sqrt(l))
            feature = rearrange(x, 'b (a d) c -> b c a d', a=h)
        elif mode == 'seq':
            b, l, c = x.shape
            h = int(math.sqrt(c))
            feature = rearrange(x, 'b c (d a) -> b c d a', a=h)
        return feature

    def _forward_reshape_tokens(self, x, mode):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [3, 3])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [3, 3])
        else:
            x = x
        if mode == 'space':
            tokens = rearrange(x, 'b c h w -> b (h w) c')
        elif mode == 'seq':
            tokens = rearrange(x, 'b c h w -> b c (h w)')
        return tokens

    def _forward_change_pha_amp(self, pha1, pha2, amp1, amp2, dim, size, mode):
        if dim == 1:
            b, c = size
            fft1 = torch.zeros([b, c, 2], dtype=torch.float).cuda()
            fft1[:, :, 0] = torch.cos(pha1.clone()) * amp2
            fft1[:, :, 1] = torch.sin(pha1.clone()) * amp2

            fft2 = torch.zeros([b, c, 2], dtype=torch.float).cuda()
            fft2[:, :, 0] = torch.cos(pha2.clone()) * amp1
            fft2[:, :, 1] = torch.sin(pha2.clone()) * amp1

            fft1 = torch.view_as_complex(fft1)
            fft2 = torch.view_as_complex(fft2)
            change1 = torch.fft.irfft(fft1, dim=-1, norm='ortho')
            change2 = torch.fft.irfft(fft2, dim=-1, norm='ortho')
            return change1, change2
        elif dim == 2:
            b, c, h, w = size
            fft1 = torch.zeros([b, c, h, w, 2], dtype=torch.float).cuda()
            fft1[:, :, :, :, 0] = torch.cos(pha1.clone()) * amp2
            fft1[:, :, :, :, 1] = torch.sin(pha1.clone()) * amp2

            fft2 = torch.zeros([b, c, h, w, 2], dtype=torch.float).cuda()
            fft2[:, :, :, :, 0] = torch.cos(pha2.clone()) * amp1
            fft2[:, :, :, :, 1] = torch.sin(pha2.clone()) * amp1

            fft1 = torch.view_as_complex(fft1)
            fft2 = torch.view_as_complex(fft2)
            if mode == 'space':
                change1 = torch.fft.irfftn(fft1, dim=(1,2,3),s=[c,h, h], norm='ortho')
                change2 = torch.fft.irfftn(fft2, dim=(1,2,3),s=[c,h, h], norm='ortho')
            elif mode == 'seq':
                change1 = torch.fft.irfft(fft1, dim=1, norm='ortho')
                change2 = torch.fft.irfft(fft2, dim=1, norm='ortho')
            return change1, change2

    def forward(self, t1input, t2input):
        # torch.autograd.set_detect_anomaly(True)

        # Encorder
        # 空[16,128,7,7],[16,128,7,4],[16,128,7,4] 谱[16,128,7,7],[16,65,7,7],[16,65,7,4]
        t1layer1 = self.layer1(t1input)
        t1layer1 = self.transformer1(t1layer1)
        t1layer1, t11amp, t11pha = self.fft_model_layer1(t1layer1)

        # [16,64,5,5],[16,64,5,3],[16,64,5,3] 谱[16,64,5,5],[16,33,5,5],[16,33,5,5]
        t1layer2 = self.layer2(t1layer1)
        t1layer2 = self.transformer2(t1layer2)
        t1layer2, t12amp, t12pha = self.fft_model_layer2(t1layer2)

        # [16,32,3,3],[16,32,3,2],[16,32,3,2] 谱[16,32,3,3],[16,17,3,3],[16,17,3,3]
        t1layer3 = self.layer3(t1layer2)
        # t1layer3 = self.transformer3(t1layer3)
        t1layer3, t13amp, t13pha = self.fft_model_layer3(t1layer3)
        # 归一化并不完整 此为空间傅里叶变化，amp为幅度

        # [16,16,1,1],[16,9],[16,9]
        t1layer4 = self.layer4(t1layer3)  # 16,16,1,1
        t1layer4, t14amp, t14pha = self.fft_model_layer4(t1layer4)
        # 归一化并不完整  此为通道傅里叶变化

        t2layer1 = self.layer1(t2input)
        t2layer1 = self.transformer1(t2layer1)
        t2layer1, t21amp, t21pha = self.fft_model_layer1(t2layer1)

        t2layer2 = self.layer2(t2layer1)
        t2layer2 = self.transformer2(t2layer2)
        t2layer2, t22amp, t22pha = self.fft_model_layer2(t2layer2)

        t2layer3 = self.layer3(t2layer2)
        # t2layer3 = self.transformer3(t2layer3)
        t2layer3, t23amp, t23pha = self.fft_model_layer3(t2layer3)

        t2layer4 = self.layer4(t2layer3)
        t2layer4, t24amp, t24pha = self.fft_model_layer4(t2layer4)

        # [16,128,7,7],[16,64,5,5],[16,32,3,3],[16,16]
        Exchange_amp_layer1_T1, Exchange_amp_layer1_T2 = self._forward_change_pha_amp(t11pha, t21pha, t11amp, t21amp, dim=2,
                                                                                      size=t11pha.size(), mode='space')
        Exchange_amp_layer2_T1, Exchange_amp_layer2_T2 = self._forward_change_pha_amp(t12pha, t22pha, t12amp, t22amp, dim=2,
                                                                                      size=t12pha.size(), mode='space')
        Exchange_amp_layer3_T1, Exchange_amp_layer3_T2 = self._forward_change_pha_amp(t13pha, t23pha, t13amp, t23amp, dim=2,
                                                                                      size=t13pha.size(), mode='space')
        Exchange_amp_layer4_T1, Exchange_amp_layer4_T2 = self._forward_change_pha_amp(t14pha, t24pha, t14amp, t24amp, dim=1,
                                                                                      size=t14pha.squeeze(-1).squeeze(-1).size(),
                                                                                      mode=None)
        # 这里是相当于同样式信息相减存异语义
        # 空间
        change_map1_1 = self._forward_reshape_tokens(Exchange_amp_layer1_T1 - t2layer1, 'space')  # [16, 49, 128]
        change_map1_2 = self._forward_reshape_tokens(t1layer1 - Exchange_amp_layer1_T2, 'space')
        change_map2_1 = self._forward_reshape_tokens(Exchange_amp_layer2_T1 - t2layer2, 'space')  # [16, 25, 64]
        change_map2_2 = self._forward_reshape_tokens(t1layer2 - Exchange_amp_layer2_T2, 'space')
        change_map3_1 = self._forward_reshape_tokens(Exchange_amp_layer3_T1 - t2layer3, 'space')  # [16, 9, 32]
        change_map3_2 = self._forward_reshape_tokens(t1layer3 - Exchange_amp_layer3_T2, 'space')
        # 光谱
        # change_map1_1 = self._forward_reshape_tokens(Exchange_amp_layer1_T1 - t2layer1, 'seq')  # [16, 128, 49]
        # change_map1_2 = self._forward_reshape_tokens(Exchange_amp_layer1_T2 - t1layer1, 'seq')
        # change_map2_1 = self._forward_reshape_tokens(Exchange_amp_layer2_T1 - t2layer2, 'seq')  # [16, 64, 25]
        # change_map2_2 = self._forward_reshape_tokens(Exchange_amp_layer2_T2 - t1layer2, 'seq')
        # change_map3_1 = self._forward_reshape_tokens(Exchange_amp_layer3_T1 - t2layer3, 'seq')  # [16, 32, 9]
        # change_map3_2 = self._forward_reshape_tokens(Exchange_amp_layer3_T2 - t1layer3, 'seq')
        change_map4_1 = Exchange_amp_layer4_T1.unsqueeze(-1).unsqueeze(-1) - t2layer4  # [16, 16,1,1]
        change_map4_2 = t1layer4 - Exchange_amp_layer4_T2.unsqueeze(-1).unsqueeze(-1)

        # 这里是相当于同语义信息相减存异样式 (N2+N1)/(N2^-N1)-(N2-N1^)=(N2^+N1^)-(N2+N1)
        change_sp_n1 = (Exchange_amp_layer1_T1 - t1layer1) - (t2layer1 - Exchange_amp_layer1_T2)  # [16, 128,7,7]
        change_sp_n2 = (Exchange_amp_layer2_T1 - t1layer2) - (t2layer2 - Exchange_amp_layer2_T2)  # [16, 64,5,5]
        change_sp_n3 = (Exchange_amp_layer3_T1 - t1layer3) - (t2layer3 - Exchange_amp_layer3_T2)  # [16. 32,3,3]
        change_sp_n4 = ((Exchange_amp_layer4_T1.unsqueeze(-1).unsqueeze(-1) - t1layer4) -
                        (t2layer4 - Exchange_amp_layer4_T2.unsqueeze(-1).unsqueeze(-1)))  # [16, 16,1,1]

        # 抛去样式信息，存留语义信息比较/抛去语义信息，存留样式信息比较
        sub_amp1 = self._forward_reshape_tokens(t21amp + t11amp, 'seq')  # [16,128,7,4]-[16,128,28]
        sub_pha1 = self._forward_reshape_tokens(t11pha - t21pha, 'space')  # [16,128,7,4]
        add_pha1 = self._forward_reshape_tokens(t11pha + t21pha, 'seq')
        # sub_pha1 = self._forward_reshape_tokens(t11pha - t21pha, 'seq')  # [16,128,28]
        sub_amp2 = self._forward_reshape_tokens(t22amp + t12amp, 'seq')  # [16,64,5,3]-[16,64,15]
        sub_pha2 = self._forward_reshape_tokens(t12pha - t22pha, 'space')  # [16,64,5,3]
        add_pha2 = self._forward_reshape_tokens(t12pha + t22pha, 'seq')
        # sub_pha2 = self._forward_reshape_tokens(t12pha - t22pha, 'seq')  # [16,64,15]

        # 对幅度进行操作(N2-N1)(是否比较激进，要不要只取一部分
        sub_amp3 = self._forward_reshape_tokens(t23amp + t13amp, 'seq')  # [16,32,3,2]-[16,32,6]
        # 对相位进行对比(A-B)(change:为变化部分,unchange:为0)
        # sub_pha3 = self._forward_reshape_tokens(t13pha - t23pha, 'space')  # [16,32,3,2]
        sub_pha3 = self._forward_reshape_tokens(t13pha - t23pha, 'space')  # [16,32,6]
        add_pha3 = self._forward_reshape_tokens(t13pha + t23pha, 'seq')

        # 同理通道
        sub_amp4 = t24amp + t14amp  # [16,9]
        # 第一批强制性语义信息
        sub_pha4 = t14pha - t24pha  # [16,9]
        add_pha4 = t14pha + t24pha

        add_pha1, add_pha2, add_pha3, add_pha4, amp1, amp2, amp3, amp4 = self.new_loss_learning(add_pha1, add_pha2,
                                                                                                add_pha3, add_pha4,                                                                                               sub_amp1, sub_amp2,
                                                                                                sub_amp3, sub_amp4)
        loss1 = self.deepinfo128(change_sp_n1, amp1, add_pha1, t1layer1+t2layer1)
        loss2 = self.deepinfo64(change_sp_n2, amp2, add_pha2, t1layer2+t2layer2)
        loss3 = self.deepinfo32(change_sp_n3, amp3, add_pha3, t1layer3+t2layer3)
        loss4 = self.deepinfo16(change_sp_n4, amp4.unsqueeze(-1).unsqueeze(-1), add_pha4.unsqueeze(-1).unsqueeze(-1),
                                t1layer4+t2layer4)
        # 空间 [16,49, 128],[16,25,64],[16,9,32],[16,16]   光谱 [16,128, 49],[16,64,25],[16,32，9],[16,16]
        # change_mask_attention_layer1 = self.transformer_decoder1_seq(sub_pha1, change_map1_1, change_map1_2)
        # change_mask_attention_layer2 = self.transformer_decoder2_seq(sub_pha2, change_map2_1, change_map2_2)
        # change_mask_attention_layer3 = self.transformer_decoder3_seq(sub_pha3, change_map3_1, change_map3_2)
        change_mask_attention_layer1 = self.transformer_decoder1(sub_pha1, change_map1_1, change_map1_2)
        change_mask_attention_layer2 = self.transformer_decoder2(sub_pha2, change_map2_1, change_map2_2)
        change_mask_attention_layer3 = self.transformer_decoder3(sub_pha3, change_map3_1, change_map3_2)
        # 一维数据处理attention
        change_mask_attention_layer4 = self.transformer_decoder4(sub_pha4, change_map4_1, change_map4_2)

        # layer1 = t1layer1 - t2layer1 + self._forward_reshape_feature(change_mask_attention_layer1, 'seq')
        # layer2 = t1layer2 - t2layer2 + self._forward_reshape_feature(change_mask_attention_layer2, 'seq')
        # layer3 = t1layer3 - t2layer3 + self._forward_reshape_feature(change_mask_attention_layer3, 'seq')
        # layer1 = self.res1(torch.cat(((t1layer1 + t2layer1 - amp1), self._forward_reshape_feature(change_mask_attention_layer1, 'space')), dim=1))
        # layer2 = self.res2(torch.cat(((t1layer2 + t2layer2 - amp2), self._forward_reshape_feature(change_mask_attention_layer2, 'space')), dim=1))
        # layer3 = self.res3(torch.cat(((t1layer3 + t2layer3 - amp3), self._forward_reshape_feature(change_mask_attention_layer3, 'space')), dim=1))
        # layer4 = self.res4(torch.cat(((t1layer4 + t2layer4 - amp4.unsqueeze(-1).unsqueeze(-1)), change_mask_attention_layer4.unsqueeze(-1).unsqueeze(-1)), dim=1))
        layer1 = (t1layer1 + t2layer1) + self._forward_reshape_feature(change_mask_attention_layer1, 'space') - amp1
        layer2 = (t1layer2 + t2layer2) + self._forward_reshape_feature(change_mask_attention_layer2, 'space') - amp2
        layer3 = (t1layer3 + t2layer3) + self._forward_reshape_feature(change_mask_attention_layer3, 'space') - amp3
        layer4 = (t1layer4 + t2layer4) + change_mask_attention_layer4.unsqueeze(-1).unsqueeze(-1) - amp4.unsqueeze(-1).unsqueeze(-1)

        # Decorder
        layer5 = self.decoder4(layer4, layer3)
        layer6 = self.decoder3(layer5, layer2)
        layer8 = self.decoder2(layer6, layer1)
        x = self.avg_pool(layer8)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, loss1, loss2, loss3, loss4
        # return x


class UNet_u(nn.Module):
    def __init__(self, args, input_size, enc_depth=1, dim_head=64):
        super().__init__()

        self.dim_head = dim_head
        self.enc_depth = enc_depth
        dim = 64
        mlp_dim = dim * 2
        self.pool_mode = None

        self.layer1 = nn.Sequential(  # 9->7 size->128
            nn.Conv2d(input_size, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(  # 7->5 128->64
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(  # 5->3 64->32
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(  # 3->1 32->16
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
    
        self.transformer1 = SequentialTransformer(patch_size=args.patches-2, channels=128,
                                                   enc_depth=enc_depth, dim_head=dim_head)
        self.transformer2 = SequentialTransformer(patch_size=args.patches-4, channels=64,
                                                   enc_depth=enc_depth, dim_head=dim_head)
        # self.transformer3 = Sequential_Transformer(patch_size=args.patches-6, channels=32,
        #                                            enc_depth=enc_depth, dim_head=dim_head)
        self.transformer_decoder1 = TransformerDecoder(dim=128, depth=1, heads=8, dim_head=64, mlp_dim=mlp_dim,
                                                      dropout=0, softmax=True, sub_pha=28, lent=49)
        self.transformer_decoder2 = TransformerDecoder(dim=64, depth=1, heads=8, dim_head=64, mlp_dim=mlp_dim,
                                                      dropout=0, softmax=True, sub_pha=15, lent=25)
        self.transformer_decoder3 = TransformerDecoder(dim=32, depth=1, heads=8, dim_head=64, mlp_dim=mlp_dim,
                                                      dropout=0, softmax=True, sub_pha=6, lent=9)

        self.transformer_decoder4 = LayerCrossAttention()
        self.fft_model_layer4 = FftModel(args=args, dim=1, shape=(1, 9, 2), learning=True, mode=None)
        self.fft_model_layer3 = FftModel(args=args, dim=2, shape=(32, 3, 2), learning=True, mode='space')
        self.fft_model_layer2 = FftModel(args=args, dim=2, shape=(64, 5, 3), learning=False, mode='space')
        self.fft_model_layer1 = FftModel(args=args, dim=2, shape=(128, 7, 4), learning=False, mode='space')

        self.decoder4 = Decoder(16, 32)
        self.decoder3 = Decoder(64, 64)
        self.decoder2 = Decoder(128, 128)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = mlp_head(dim=256)  # 分类器
        self.classifier_pha = mlp_head(dim=16)
        self.sigmoid = nn.Sigmoid()

    def _forward_reshape_feature(self, x, mode):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [3, 3])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [3, 3])
        else:
            x = x
        if mode == 'space':
            b, l, c = x.shape
            h = int(math.sqrt(l))
            feature = rearrange(x, 'b (a d) c -> b c a d', a=h)
        elif mode == 'seq':
            b, l, c = x.shape
            h = int(math.sqrt(c))
            feature = rearrange(x, 'b c (d a) -> b c d a', a=h)
        return feature

    def _forward_reshape_tokens(self, x, mode):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [3, 3])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [3, 3])
        else:
            x = x
        if mode == 'space':
            tokens = rearrange(x, 'b c h w -> b (h w) c')
        elif mode == 'seq':
            tokens = rearrange(x, 'b c h w -> b c (h w)')
        return tokens

    def _forward_change_pha_amp(self, pha1, pha2, amp1, amp2, dim, size, mode):
        if dim == 1:
            b, c = size
            fft1 = torch.zeros([b, c, 2], dtype=torch.float).cuda()
            fft1[:, :, 0] = torch.cos(pha1.clone()) * amp2
            fft1[:, :, 1] = torch.sin(pha1.clone()) * amp2

            fft2 = torch.zeros([b, c, 2], dtype=torch.float).cuda()
            fft2[:, :, 0] = torch.cos(pha2.clone()) * amp1
            fft2[:, :, 1] = torch.sin(pha2.clone()) * amp1

            fft1 = torch.view_as_complex(fft1)
            fft2 = torch.view_as_complex(fft2)
            change1 = torch.fft.irfft(fft1, dim=-1, norm='ortho')
            change2 = torch.fft.irfft(fft2, dim=-1, norm='ortho')
            return change1, change2
        elif dim == 2:
            b, c, h, w = size
            fft1 = torch.zeros([b, c, h, w, 2], dtype=torch.float).cuda()
            fft1[:, :, :, :, 0] = torch.cos(pha1.clone()) * amp2
            fft1[:, :, :, :, 1] = torch.sin(pha1.clone()) * amp2

            fft2 = torch.zeros([b, c, h, w, 2], dtype=torch.float).cuda()
            fft2[:, :, :, :, 0] = torch.cos(pha2.clone()) * amp1
            fft2[:, :, :, :, 1] = torch.sin(pha2.clone()) * amp1

            fft1 = torch.view_as_complex(fft1)
            fft2 = torch.view_as_complex(fft2)
            if mode == 'space':
                change1 = torch.fft.irfftn(fft1, dim=(1,2,3),s=[c,h, h], norm='ortho')
                change2 = torch.fft.irfftn(fft2, dim=(1,2,3),s=[c,h, h], norm='ortho')
            elif mode == 'seq':
                change1 = torch.fft.irfft(fft1, dim=1, norm='ortho')
                change2 = torch.fft.irfft(fft2, dim=1, norm='ortho')
            return change1, change2

    def forward(self, t1input, t2input):
        # torch.autograd.set_detect_anomaly(True)

        # Encorder
        # 空[16,128,7,7],[16,128,7,4],[16,128,7,4] 谱[16,128,7,7],[16,65,7,7],[16,65,7,4]
        t1layer1 = self.layer1(t1input)
        t1layer1 = self.transformer1(t1layer1)
        t1layer1, t11amp, t11pha = self.fft_model_layer1(t1layer1)

        # [16,64,5,5],[16,64,5,3],[16,64,5,3] 谱[16,64,5,5],[16,33,5,5],[16,33,5,5]
        t1layer2 = self.layer2(t1layer1)
        t1layer2 = self.transformer2(t1layer2)
        t1layer2, t12amp, t12pha = self.fft_model_layer2(t1layer2)

        # [16,32,3,3],[16,32,3,2],[16,32,3,2] 谱[16,32,3,3],[16,17,3,3],[16,17,3,3]
        t1layer3 = self.layer3(t1layer2)
        # t1layer3 = self.transformer3(t1layer3)
        t1layer3, t13amp, t13pha = self.fft_model_layer3(t1layer3)
        # 归一化并不完整 此为空间傅里叶变化，amp为幅度

        # [16,16,1,1],[16,9],[16,9]
        t1layer4 = self.layer4(t1layer3)  # 16,16,1,1
        t1layer4, t14amp, t14pha = self.fft_model_layer4(t1layer4)
        # 归一化并不完整  此为通道傅里叶变化

        t2layer1 = self.layer1(t2input)
        t2layer1 = self.transformer1(t2layer1)
        t2layer1, t21amp, t21pha = self.fft_model_layer1(t2layer1)

        t2layer2 = self.layer2(t2layer1)
        t2layer2 = self.transformer2(t2layer2)
        t2layer2, t22amp, t22pha = self.fft_model_layer2(t2layer2)

        t2layer3 = self.layer3(t2layer2)
        # t2layer3 = self.transformer3(t2layer3)
        t2layer3, t23amp, t23pha = self.fft_model_layer3(t2layer3)

        t2layer4 = self.layer4(t2layer3)
        t2layer4, t24amp, t24pha = self.fft_model_layer4(t2layer4)

        # [16,128,7,7],[16,64,5,5],[16,32,3,3],[16,16]
        Exchange_amp_layer1_T1, Exchange_amp_layer1_T2 = self._forward_change_pha_amp(t11pha, t21pha, t11amp, t21amp, dim=2,
                                                                                      size=t11pha.size(), mode='space')
        Exchange_amp_layer2_T1, Exchange_amp_layer2_T2 = self._forward_change_pha_amp(t12pha, t22pha, t12amp, t22amp, dim=2,
                                                                                      size=t12pha.size(), mode='space')
        Exchange_amp_layer3_T1, Exchange_amp_layer3_T2 = self._forward_change_pha_amp(t13pha, t23pha, t13amp, t23amp, dim=2,
                                                                                      size=t13pha.size(), mode='space')
        Exchange_amp_layer4_T1, Exchange_amp_layer4_T2 = self._forward_change_pha_amp(t14pha, t24pha, t14amp, t24amp, dim=1,
                                                                                      size=t14pha.squeeze(-1).squeeze(-1).size(),
                                                                                      mode=None)
        # 这里是相当于同样式信息相减存异语义
        # 空间
        change_map1_1 = self._forward_reshape_tokens(Exchange_amp_layer1_T1 - t2layer1, 'space')  # [16, 49, 128]
        change_map1_2 = self._forward_reshape_tokens(t1layer1 - Exchange_amp_layer1_T2, 'space')
        change_map2_1 = self._forward_reshape_tokens(Exchange_amp_layer2_T1 - t2layer2, 'space')  # [16, 25, 64]
        change_map2_2 = self._forward_reshape_tokens(t1layer2 - Exchange_amp_layer2_T2, 'space')
        change_map3_1 = self._forward_reshape_tokens(Exchange_amp_layer3_T1 - t2layer3, 'space')  # [16, 9, 32]
        change_map3_2 = self._forward_reshape_tokens(t1layer3 - Exchange_amp_layer3_T2, 'space')
        change_map4_1 = Exchange_amp_layer4_T1.unsqueeze(-1).unsqueeze(-1) - t2layer4  # [16, 16,1,1]
        change_map4_2 = t1layer4 - Exchange_amp_layer4_T2.unsqueeze(-1).unsqueeze(-1)


        # 抛去样式信息，存留语义信息比较/抛去语义信息，存留样式信息比较
        sub_pha1 = self._forward_reshape_tokens(t11pha - t21pha, 'space')  # [16,128,7,4]
        sub_pha2 = self._forward_reshape_tokens(t12pha - t22pha, 'space')  # [16,64,5,3]
        sub_pha3 = self._forward_reshape_tokens(t13pha - t23pha, 'space')  # [16,32,6]
        # 第一批强制性语义信息
        sub_pha4 = t14pha - t24pha  # [16,9]


        change_mask_attention_layer1 = self.transformer_decoder1(sub_pha1, change_map1_1, change_map1_2)
        change_mask_attention_layer2 = self.transformer_decoder2(sub_pha2, change_map2_1, change_map2_2)
        change_mask_attention_layer3 = self.transformer_decoder3(sub_pha3, change_map3_1, change_map3_2)
        # 一维数据处理attention
        change_mask_attention_layer4 = self.transformer_decoder4(sub_pha4, change_map4_1, change_map4_2)

        # layer1 = t1layer1 - t2layer1 + self._forward_reshape_feature(change_mask_attention_layer1, 'seq')
        # layer2 = t1layer2 - t2layer2 + self._forward_reshape_feature(change_mask_attention_layer2, 'seq')
        # layer3 = t1layer3 - t2layer3 + self._forward_reshape_feature(change_mask_attention_layer3, 'seq')
        # layer1 = self.res1(torch.cat(((t1layer1 + t2layer1 - amp1), self._forward_reshape_feature(change_mask_attention_layer1, 'space')), dim=1))
        # layer2 = self.res2(torch.cat(((t1layer2 + t2layer2 - amp2), self._forward_reshape_feature(change_mask_attention_layer2, 'space')), dim=1))
        # layer3 = self.res3(torch.cat(((t1layer3 + t2layer3 - amp3), self._forward_reshape_feature(change_mask_attention_layer3, 'space')), dim=1))
        # layer4 = self.res4(torch.cat(((t1layer4 + t2layer4 - amp4.unsqueeze(-1).unsqueeze(-1)), change_mask_attention_layer4.unsqueeze(-1).unsqueeze(-1)), dim=1))
        layer1 = (t1layer1 + t2layer1) + self._forward_reshape_feature(change_mask_attention_layer1, 'space')
        layer2 = (t1layer2 + t2layer2) + self._forward_reshape_feature(change_mask_attention_layer2, 'space')
        layer3 = (t1layer3 + t2layer3) + self._forward_reshape_feature(change_mask_attention_layer3, 'space')
        layer4 = (t1layer4 + t2layer4) + change_mask_attention_layer4.unsqueeze(-1).unsqueeze(-1)

        # Decorder
        layer5 = self.decoder4(layer4, layer3)
        layer6 = self.decoder3(layer5, layer2)
        layer8 = self.decoder2(layer6, layer1)
        x = self.avg_pool(layer8)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class UNet_s(nn.Module):
    def __init__(self, args, input_size, enc_depth=1, dim_head=64):
        super().__init__()

        self.dim_head = dim_head
        self.enc_depth = enc_depth
        dim = 64
        mlp_dim = dim * 2
        self.pool_mode = None

        self.layer1 = nn.Sequential(  # 9->7 size->128
            nn.Conv2d(input_size, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(  # 7->5 128->64
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(  # 5->3 64->32
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(  # 3->1 32->16
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.transformer1 = SequentialTransformer(patch_size=args.patches - 2, channels=128,
                                                  enc_depth=enc_depth, dim_head=dim_head)
        self.transformer2 = SequentialTransformer(patch_size=args.patches - 4, channels=64,
                                                  enc_depth=enc_depth, dim_head=dim_head)
        # self.transformer3 = Sequential_Transformer(patch_size=args.patches-6, channels=32,
        #                                            enc_depth=enc_depth, dim_head=dim_head)

        self.fft_model_layer4 = FftModel(args=args, dim=1, shape=(1, 9, 2), learning=True, mode=None)
        self.fft_model_layer3 = FftModel(args=args, dim=2, shape=(32, 3, 2), learning=True, mode='space')
        self.fft_model_layer2 = FftModel(args=args, dim=2, shape=(64, 5, 3), learning=False, mode='space')
        self.fft_model_layer1 = FftModel(args=args, dim=2, shape=(128, 7, 4), learning=False, mode='space')

        self.decoder4 = Decoder(16, 32)
        self.decoder3 = Decoder(64, 64)
        self.decoder2 = Decoder(128, 128)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = mlp_head(dim=256)  # 分类器
        self.classifier_pha = mlp_head(dim=16)
        self.sigmoid = nn.Sigmoid()

    def _forward_reshape_feature(self, x, mode):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [3, 3])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [3, 3])
        else:
            x = x
        if mode == 'space':
            b, l, c = x.shape
            h = int(math.sqrt(l))
            feature = rearrange(x, 'b (a d) c -> b c a d', a=h)
        elif mode == 'seq':
            b, l, c = x.shape
            h = int(math.sqrt(c))
            feature = rearrange(x, 'b c (d a) -> b c d a', a=h)
        return feature

    def _forward_reshape_tokens(self, x, mode):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [3, 3])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [3, 3])
        else:
            x = x
        if mode == 'space':
            tokens = rearrange(x, 'b c h w -> b (h w) c')
        elif mode == 'seq':
            tokens = rearrange(x, 'b c h w -> b c (h w)')
        return tokens



    def forward(self, t1input, t2input):
        # torch.autograd.set_detect_anomaly(True)

        # Encorder
        # 空[16,128,7,7],[16,128,7,4],[16,128,7,4] 谱[16,128,7,7],[16,65,7,7],[16,65,7,4]
        t1layer1 = self.layer1(t1input)
        t1layer1 = self.transformer1(t1layer1)
        t1layer1, t11amp, t11pha = self.fft_model_layer1(t1layer1)

        # [16,64,5,5],[16,64,5,3],[16,64,5,3] 谱[16,64,5,5],[16,33,5,5],[16,33,5,5]
        t1layer2 = self.layer2(t1layer1)
        t1layer2 = self.transformer2(t1layer2)
        t1layer2, t12amp, t12pha = self.fft_model_layer2(t1layer2)

        # [16,32,3,3],[16,32,3,2],[16,32,3,2] 谱[16,32,3,3],[16,17,3,3],[16,17,3,3]
        t1layer3 = self.layer3(t1layer2)
        # t1layer3 = self.transformer3(t1layer3)
        t1layer3, t13amp, t13pha = self.fft_model_layer3(t1layer3)
        # 归一化并不完整 此为空间傅里叶变化，amp为幅度

        # [16,16,1,1],[16,9],[16,9]
        t1layer4 = self.layer4(t1layer3)  # 16,16,1,1
        t1layer4, t14amp, t14pha = self.fft_model_layer4(t1layer4)
        # 归一化并不完整  此为通道傅里叶变化

        t2layer1 = self.layer1(t2input)
        t2layer1 = self.transformer1(t2layer1)
        t2layer1, t21amp, t21pha = self.fft_model_layer1(t2layer1)

        t2layer2 = self.layer2(t2layer1)
        t2layer2 = self.transformer2(t2layer2)
        t2layer2, t22amp, t22pha = self.fft_model_layer2(t2layer2)

        t2layer3 = self.layer3(t2layer2)
        # t2layer3 = self.transformer3(t2layer3)
        t2layer3, t23amp, t23pha = self.fft_model_layer3(t2layer3)

        t2layer4 = self.layer4(t2layer3)
        t2layer4, t24amp, t24pha = self.fft_model_layer4(t2layer4)



        layer1 = (t1layer1 + t2layer1)
        layer2 = (t1layer2 + t2layer2)
        layer3 = (t1layer3 + t2layer3)
        layer4 = (t1layer4 + t2layer4)

        # Decorder
        layer5 = self.decoder4(layer4, layer3)
        layer6 = self.decoder3(layer5, layer2)
        layer8 = self.decoder2(layer6, layer1)
        x = self.avg_pool(layer8)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class UNet_f(nn.Module):
    def __init__(self, args, input_size, enc_depth=1, dim_head=64):
        super().__init__()

        self.dim_head = dim_head
        self.enc_depth = enc_depth
        dim = 64
        mlp_dim = dim * 2
        self.pool_mode = None

        self.layer1 = nn.Sequential(  # 9->7 size->128
            nn.Conv2d(input_size, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(  # 7->5 128->64
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(  # 5->3 64->32
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(  # 3->1 32->16
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.transformer1 = SequentialTransformer(patch_size=args.patches - 2, channels=128,
                                                  enc_depth=enc_depth, dim_head=dim_head)
        self.transformer2 = SequentialTransformer(patch_size=args.patches - 4, channels=64,
                                                  enc_depth=enc_depth, dim_head=dim_head)

        self.decoder4 = Decoder(16, 32)
        self.decoder3 = Decoder(64, 64)
        self.decoder2 = Decoder(128, 128)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = mlp_head(dim=256)  # 分类器
        self.classifier_pha = mlp_head(dim=16)
        self.sigmoid = nn.Sigmoid()

    def _forward_reshape_feature(self, x, mode):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [3, 3])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [3, 3])
        else:
            x = x
        if mode == 'space':
            b, l, c = x.shape
            h = int(math.sqrt(l))
            feature = rearrange(x, 'b (a d) c -> b c a d', a=h)
        elif mode == 'seq':
            b, l, c = x.shape
            h = int(math.sqrt(c))
            feature = rearrange(x, 'b c (d a) -> b c d a', a=h)
        return feature

    def _forward_reshape_tokens(self, x, mode):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [3, 3])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [3, 3])
        else:
            x = x
        if mode == 'space':
            tokens = rearrange(x, 'b c h w -> b (h w) c')
        elif mode == 'seq':
            tokens = rearrange(x, 'b c h w -> b c (h w)')
        return tokens

    def forward(self, t1input, t2input):
        # torch.autograd.set_detect_anomaly(True)

        # Encorder
        # 空[16,128,7,7],[16,128,7,4],[16,128,7,4] 谱[16,128,7,7],[16,65,7,7],[16,65,7,4]
        t1layer1 = self.layer1(t1input)
        t1layer1 = self.transformer1(t1layer1)

        # [16,64,5,5],[16,64,5,3],[16,64,5,3] 谱[16,64,5,5],[16,33,5,5],[16,33,5,5]
        t1layer2 = self.layer2(t1layer1)
        t1layer2 = self.transformer2(t1layer2)


        # [16,32,3,3],[16,32,3,2],[16,32,3,2] 谱[16,32,3,3],[16,17,3,3],[16,17,3,3]
        t1layer3 = self.layer3(t1layer2)
        # t1layer3 = self.transformer3(t1layer3)

        # 归一化并不完整 此为空间傅里叶变化，amp为幅度

        # [16,16,1,1],[16,9],[16,9]
        t1layer4 = self.layer4(t1layer3)  # 16,16,1,1

        # 归一化并不完整  此为通道傅里叶变化

        t2layer1 = self.layer1(t2input)
        t2layer1 = self.transformer1(t2layer1)


        t2layer2 = self.layer2(t2layer1)
        t2layer2 = self.transformer2(t2layer2)


        t2layer3 = self.layer3(t2layer2)


        t2layer4 = self.layer4(t2layer3)


        layer1 = (t1layer1 + t2layer1)
        layer2 = (t1layer2 + t2layer2)
        layer3 = (t1layer3 + t2layer3)
        layer4 = (t1layer4 + t2layer4)

        # Decorder
        layer5 = self.decoder4(layer4, layer3)
        layer6 = self.decoder3(layer5, layer2)
        layer8 = self.decoder2(layer6, layer1)
        x = self.avg_pool(layer8)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x