import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from pytorch_utils import do_mixup, interpolate, pad_framewise_output
from quaternion_layers import QuaternionConv2d ,QuaternionConv2dNoBias, QuaternionConv, QuaternionLinear
from torchinfo import summary
import torchaudio
import numpy as np
import os 


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class QConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(QConvBlock, self).__init__()
        
        self.conv1 = QuaternionConv2dNoBias(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = QuaternionConv2dNoBias(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        # init_layer(self.conv1)
        # init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class QConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(QConvBlock5x5, self).__init__()
        
        self.conv1 = QuaternionConv2dNoBias(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        # init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()
        
        self.activation = activation
        self.temperature = temperature
        self.att = QuaternionConv(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = QuaternionConv(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()
        
    def init_weights(self):
        # init_layer(self.att)
        # init_layer(self.cla)
        init_bn(self.bn_att)
         
    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)





class QConvBlock_pruned(nn.Module):
    def __init__(self, in_channels_1, out_channels_1,out_channels_2):
       
        super(QConvBlock_pruned, self).__init__()
       
        self.conv1 = QuaternionConv2dNoBias(in_channels=in_channels_1, 
                              out_channels=out_channels_1,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
                             
        self.conv2 = QuaternionConv2dNoBias(in_channels=out_channels_1, 
                              out_channels=out_channels_2,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
                             
        self.bn1 = nn.BatchNorm2d(out_channels_1)
        self.bn2 = nn.BatchNorm2d(out_channels_2)

        self.init_weight()
       
    def init_weight(self):
        # init_layer(self.conv1)
        # init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

       
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
       
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
       
        return x



############################################# QCNN14 Model #######################################
class QCnn14_train(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(QCnn14_train, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = QConvBlock(in_channels=4, out_channels=64)
        self.conv_block2 = QConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = QConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = QConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = QConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = QConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = QuaternionLinear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        # init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        print("input shape", input.shape)
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        
        
        
        x_first = torchaudio.functional.compute_deltas(x)
        x_second = torchaudio.functional.compute_deltas(x_first)
        x_third = torchaudio.functional.compute_deltas(x_second) 
        #quternionic converter
        x_quaternion = torch.cat([x,x_first, x_second, x_third], dim=1) # (batch_size, 4, time_steps, freq_bins)
        #print("spectrogram shape", x_quaternion.shape)
        x = self.logmel_extractor(x_quaternion)    # (batch_size, 4, time_steps, mel_bins)
        #print("logmel shape", x.shape)

        x = x.transpose(1, 3)
        #print("logmel shape after first transpose", x.shape)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        #print(" Shape after bachnorm normalisation transpose", x.shape)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
    



############################################ RESNET Models #######################################

def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return QuaternionConv2dNoBias(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, dilatation=1,bias=False)

def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return QuaternionConv2dNoBias(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        # init_layer(self.conv1)
        init_bn(self.bn1)
        # init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
    

class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        # init_layer(self.conv1)
        init_bn(self.bn1)
        # init_layer(self.conv2)
        init_bn(self.bn2)
        # init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                # init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                # init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% QCNN14 pruned models ############################

class QCnn14_pruned(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
            fmax, classes_num):#, pooling_type, pooling_factor):
        import os
        import numpy as np

        super(QCnn14_pruned, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        from collections import OrderedDict
        path = './QCNN_sorted_Index/OP/' # path_to_sorted_indexex (add here)

        p = 0.75   # pruning ratio (change (0.25,0.50,0.75))
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
        p5 = 0
        p6 = 0
        p7 = p
        p8 = p
        p9 = p
        p10 = p
        p11 = p
        p12 = p

        C1_r_b1 = sorted(np.load(os.path.join(path,'conv_block1.conv1_mean_score.npy'))[int(16*p1):16])
        C1_i_b1 = sorted(np.load(os.path.join(path,'conv_block1.conv1_mean_score.npy'))[int(16*p1):16])
        C1_j_b1 = sorted(np.load(os.path.join(path,'conv_block1.conv1_mean_score.npy'))[int(16*p1):16])
        C1_k_b1 = sorted(np.load(os.path.join(path,'conv_block1.conv1_mean_score.npy'))[int(16*p1):16])
        # C1_k = np.arange(0,100)[16+int(16*p1):32]
        # 
        C2_r_b1= sorted(np.load(os.path.join(path,'conv_block1.conv2_mean_score.npy'))[int(16*p2):16])
        C2_i_b1= sorted(np.load(os.path.join(path,'conv_block1.conv2_mean_score.npy'))[int(16*p2):16])
        C2_j_b1 = sorted(np.load(os.path.join(path,'conv_block1.conv2_mean_score.npy'))[int(16*p2):16])
        C2_k_b1 = sorted(np.load(os.path.join(path,'conv_block1.conv2_mean_score.npy'))[int(16*p2):16])
        
        
        #'''''''''''''''''''''''''''''' block 2
        C1_r_b2 = sorted(np.load(os.path.join(path,'conv_block2.conv1_mean_score.npy'))[int(32*p3):32])
        C1_i_b2 = sorted(np.load(os.path.join(path,'conv_block2.conv1_mean_score.npy'))[int(32*p3):32])
        C1_j_b2 = sorted(np.load(os.path.join(path,'conv_block2.conv1_mean_score.npy'))[int(32*p3):32])
        C1_k_b2 = sorted(np.load(os.path.join(path,'conv_block2.conv1_mean_score.npy'))[int(32*p3):32])
        
        
        
        #sorted(np.load(os.path.join(path,'conv_block2.conv1.weight.npy'))[int(128*p3):128])
        
        C2_r_b2 = sorted(np.load(os.path.join(path,'conv_block2.conv2_mean_score.npy'))[int(32*p4):32])
        C2_i_b2 = sorted(np.load(os.path.join(path,'conv_block2.conv2_mean_score.npy'))[int(32*p4):32])
        C2_j_b2 = sorted(np.load(os.path.join(path,'conv_block2.conv2_mean_score.npy'))[int(32*p4):32])
        C2_k_b2 = sorted(np.load(os.path.join(path,'conv_block2.conv2_mean_score.npy'))[int(32*p4):32])
        
        #''''''''''''''''''''''''''''''' block 3
        
        
        #sorted(np.load(os.path.join(path,'conv_block2.conv2.weight.npy'))[int(128*p4):128])
        
        
        C1_r_b3 = sorted(np.load(os.path.join(path,'conv_block3.conv1_mean_score.npy'))[int(64*p5):64])
        C1_i_b3 = sorted(np.load(os.path.join(path,'conv_block3.conv1_mean_score.npy'))[int(64*p5):64])
        C1_j_b3 = sorted(np.load(os.path.join(path,'conv_block3.conv1_mean_score.npy'))[int(64*p5):64])
        C1_k_b3 = sorted(np.load(os.path.join(path,'conv_block3.conv1_mean_score.npy'))[int(64*p5):64])
        
        
        #sorted(np.load(os.path.join(path,'conv_block3.conv1.weight.npy'))[int(256*p5):256])
        C2_r_b3 = sorted(np.load(os.path.join(path,'conv_block3.conv2_mean_score.npy'))[int(64*p6):64])
        C2_i_b3 = sorted(np.load(os.path.join(path,'conv_block3.conv2_mean_score.npy'))[int(64*p6):64])
        C2_j_b3 = sorted(np.load(os.path.join(path,'conv_block3.conv2_mean_score.npy'))[int(64*p6):64])
        C2_k_b3 = sorted(np.load(os.path.join(path,'conv_block3.conv2_mean_score.npy'))[int(64*p6):64])
        
        #sorted(np.load(os.path.join(path,'conv_block3.conv2.weight.npy'))[int(256*p6):256])
        
        
        #'''''''''''''''''''''''''''''''''''''''''block 4
        #sorted(np.load(os.path.join(path,'conv_block4.conv2.weight.npy'))[int(512*p8):512])
        
        C1_r_b4 = sorted(np.load(os.path.join(path,'conv_block4.conv1_mean_score.npy'))[int(128*p7):128])
        C1_i_b4 = sorted(np.load(os.path.join(path,'conv_block4.conv1_mean_score.npy'))[int(128*p7):128])
        C1_j_b4 = sorted(np.load(os.path.join(path,'conv_block4.conv1_mean_score.npy'))[int(128*p7):128])
        C1_k_b4 = sorted(np.load(os.path.join(path,'conv_block4.conv1_mean_score.npy'))[int(128*p7):128])
        
        
        #sorted(np.load(os.path.join(path,'conv_block4.conv1.weight.npy'))[int(512*p7):512])
        C2_r_b4 = sorted(np.load(os.path.join(path,'conv_block4.conv2_mean_score.npy'))[int(128*p8):128])
        C2_i_b4 = sorted(np.load(os.path.join(path,'conv_block4.conv2_mean_score.npy'))[int(128*p8):128])
        C2_j_b4 = sorted(np.load(os.path.join(path,'conv_block4.conv2_mean_score.npy'))[int(128*p8):128])
        C2_k_b4 = sorted(np.load(os.path.join(path,'conv_block4.conv2_mean_score.npy'))[int(128*p8):128])
        
        
        #'''''''''''''''''''''''''''''''''''block 5
        
        C1_r_b5 = sorted(np.load(os.path.join(path,'conv_block5.conv1_mean_score.npy'))[int(256*p9):256])
        C1_i_b5 = sorted(np.load(os.path.join(path,'conv_block5.conv1_mean_score.npy'))[int(256*p9):256])
        C1_j_b5 = sorted(np.load(os.path.join(path,'conv_block5.conv1_mean_score.npy'))[int(256*p9):256])
        C1_k_b5 = sorted(np.load(os.path.join(path,'conv_block5.conv1_mean_score.npy'))[int(256*p9):256])
        
        
        C2_r_b5 = sorted(np.load(os.path.join(path,'conv_block5.conv2_mean_score.npy'))[int(256*p10):256])
        C2_i_b5 = sorted(np.load(os.path.join(path,'conv_block5.conv2_mean_score.npy'))[int(256*p10):256])
        C2_j_b5 = sorted(np.load(os.path.join(path,'conv_block5.conv2_mean_score.npy'))[int(256*p10):256])
        C2_k_b5 = sorted(np.load(os.path.join(path,'conv_block5.conv2_mean_score.npy'))[int(256*p10):256])
        
        
        #'''''''''''''''''''''''''''''''''''''block 6
        ##sorted(np.load(os.path.join(path,'conv_block5.conv2.weight.npy'))[int(1024*p10):1024])
        
        C1_r_b6 = sorted(np.load(os.path.join(path,'conv_block6.conv1_mean_score.npy'))[int(512*p11):512])
        C1_i_b6 = sorted(np.load(os.path.join(path,'conv_block6.conv1_mean_score.npy'))[int(512*p11):512])
        C1_j_b6 = sorted(np.load(os.path.join(path,'conv_block6.conv1_mean_score.npy'))[int(512*p11):512])
        C1_k_b6 = sorted(np.load(os.path.join(path,'conv_block6.conv1_mean_score.npy'))[int(512*p11):512])
        
        
        
        #sorted(np.load(os.path.join(path,'conv_block6.conv1.weight.npy'))[int(2048*p11):2048])
        C2_r_b6 = sorted(np.load(os.path.join(path,'conv_block6.conv2_mean_score.npy'))[int(512*p12):512])
        C2_i_b6 = sorted(np.load(os.path.join(path,'conv_block6.conv2_mean_score.npy'))[int(512*p12):512])
        C2_j_b6 = sorted(np.load(os.path.join(path,'conv_block6.conv2_mean_score.npy'))[int(512*p12):512])
        C2_k_b6 = sorted(np.load(os.path.join(path,'conv_block6.conv2_mean_score.npy'))[int(512*p12):512])
                ##sorted(np.load(os.path.join(path,'conv_block6.conv2.weight.npy'))[int(2048*p12):2048])


        conv_index = OrderedDict()
        
        conv_index['conv_block1.conv1.r_weight'] = C1_r_b1
        conv_index['conv_block1.conv1.i_weight'] = C1_i_b1
        conv_index['conv_block1.conv1.j_weight'] = C1_j_b1
        conv_index['conv_block1.conv1.k_weight'] = C1_k_b1
        
        conv_index['conv_block1.conv2.r_weight'] = C2_r_b1
        conv_index['conv_block1.conv2.i_weight'] = C2_i_b1
        conv_index['conv_block1.conv2.j_weight'] = C2_j_b1
        conv_index['conv_block1.conv2.k_weight'] = C2_k_b1
        
        conv_index['conv_block2.conv1.r_weight'] = C1_r_b2
        conv_index['conv_block2.conv1.i_weight'] = C1_i_b2
        conv_index['conv_block2.conv1.j_weight'] = C1_j_b2
        conv_index['conv_block2.conv1.k_weight'] = C1_k_b2
        
        conv_index['conv_block2.conv2.r_weight'] = C2_r_b2
        conv_index['conv_block2.conv2.i_weight'] = C2_i_b2
        conv_index['conv_block2.conv2.j_weight'] = C2_j_b2
        conv_index['conv_block2.conv2.k_weight'] = C2_k_b2
        
        
        
        conv_index['conv_block3.conv1.r_weight'] = C1_r_b3
        conv_index['conv_block3.conv1.i_weight'] = C1_i_b3
        conv_index['conv_block3.conv1.j_weight'] = C1_j_b3
        conv_index['conv_block3.conv1.k_weight'] = C1_k_b3
        
        conv_index['conv_block3.conv2.r_weight'] = C2_r_b3
        conv_index['conv_block3.conv2.i_weight'] = C2_i_b3
        conv_index['conv_block3.conv2.j_weight'] = C2_j_b3
        conv_index['conv_block3.conv2.k_weight'] = C2_k_b3
        
        
        
        conv_index['conv_block4.conv1.r_weight'] = C1_r_b4
        conv_index['conv_block4.conv1.i_weight'] = C1_i_b4
        conv_index['conv_block4.conv1.j_weight'] = C1_j_b4
        conv_index['conv_block4.conv1.k_weight'] = C1_k_b4
        
        conv_index['conv_block4.conv2.r_weight'] = C2_r_b4
        conv_index['conv_block4.conv2.i_weight'] = C2_i_b4
        conv_index['conv_block4.conv2.j_weight'] = C2_j_b4
        conv_index['conv_block4.conv2.k_weight'] = C2_k_b4
        
        
        
        conv_index['conv_block5.conv1.r_weight'] = C1_r_b5
        conv_index['conv_block5.conv1.i_weight'] = C1_i_b5
        conv_index['conv_block5.conv1.j_weight'] = C1_j_b5
        conv_index['conv_block5.conv1.k_weight'] = C1_k_b5
        
        conv_index['conv_block5.conv2.r_weight'] = C2_r_b5
        conv_index['conv_block5.conv2.i_weight'] = C2_i_b5
        conv_index['conv_block5.conv2.j_weight'] = C2_j_b5
        conv_index['conv_block5.conv2.k_weight'] = C2_k_b5
        
        conv_index['conv_block6.conv1.r_weight'] = C1_r_b6
        conv_index['conv_block6.conv1.i_weight'] = C1_i_b6
        conv_index['conv_block6.conv1.j_weight'] = C1_j_b6
        conv_index['conv_block6.conv1.k_weight'] = C1_k_b6
        
        conv_index['conv_block6.conv2.r_weight'] = C2_r_b6
        conv_index['conv_block6.conv2.i_weight'] = C2_i_b6
        conv_index['conv_block6.conv2.j_weight'] = C2_j_b6
        conv_index['conv_block6.conv2.k_weight'] = C2_k_b6


        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = QConvBlock_pruned(in_channels_1= 4 ,  out_channels_1=int(64*(1-p1)),out_channels_2=int(64*(1-p2)))
        self.conv_block2 = QConvBlock_pruned(in_channels_1=int(64*(1-p2)), out_channels_1=int(128*(1-p3)),out_channels_2=int(128*(1-p4)))
        self.conv_block3 = QConvBlock_pruned(in_channels_1=int(128*(1-p4)), out_channels_1=int(256*(1-p5)),out_channels_2=int(256*(1-p6)))
        self.conv_block4 = QConvBlock_pruned(in_channels_1=int(256*(1-p6)), out_channels_1=int(512*(1-p7)),out_channels_2=int(512*(1-p8)))
        self.conv_block5 = QConvBlock_pruned(in_channels_1=int(512*(1-p8)), out_channels_1=int(1024*(1-p9)),out_channels_2=int(1024*(1-p10)))
        self.conv_block6 = QConvBlock_pruned(in_channels_1=int(1024*(1-p10)), out_channels_1=int((1-p11)*2048),out_channels_2=int(2048*(1-p12)))

        self.fc1 = QuaternionLinear(int(2048*(1-p12)), 2048, bias=True) #nn.Linear(int(2048*(1-p12)), 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        # self.init_weight()
        checkpoint_path_qcnn14 = './980000_iterations.pth'
        checkpoint = torch.load(checkpoint_path_qcnn14, map_location=torch.device('cpu'))

        weights = checkpoint['model'] #['model']
        weights_pruned = weights #checkpoint['model'] #['model']


        # conv_key_list = ['conv_block1.conv1.weight', 'conv_block1.conv2.weight','conv_block2.conv1.weight', 'conv_block2.conv2.weight','conv_block3.conv1.weight', 'conv_block3.conv2.weight','conv_block4.conv1.weight', 'conv_block4.conv2.weight','conv_block5.conv1.weight', 'conv_block5.conv2.weight','conv_block6.conv1.weight', 'conv_block6.conv2.weight']
        bn_key_list = ['conv_block1.bn1.weight', 'conv_block1.bn1.bias', 'conv_block1.bn1.running_mean', 'conv_block1.bn1.running_var','conv_block1.bn2.weight', 'conv_block1.bn2.bias', 'conv_block1.bn2.running_mean', 'conv_block1.bn2.running_var','conv_block2.bn1.weight', 'conv_block2.bn1.bias', 'conv_block2.bn1.running_mean', 'conv_block2.bn1.running_var','conv_block2.bn2.weight', 'conv_block2.bn2.bias', 'conv_block2.bn2.running_mean', 'conv_block2.bn2.running_var','conv_block3.bn1.weight', 'conv_block3.bn1.bias', 'conv_block3.bn1.running_mean', 'conv_block3.bn1.running_var','conv_block3.bn2.weight', 'conv_block3.bn2.bias', 'conv_block3.bn2.running_mean', 'conv_block3.bn2.running_var','conv_block4.bn1.weight', 'conv_block4.bn1.bias', 'conv_block4.bn1.running_mean', 'conv_block4.bn1.running_var','conv_block4.bn2.weight', 'conv_block4.bn2.bias', 'conv_block4.bn2.running_mean', 'conv_block4.bn2.running_var','conv_block5.bn1.weight', 'conv_block5.bn1.bias', 'conv_block5.bn1.running_mean', 'conv_block5.bn1.running_var','conv_block5.bn2.weight', 'conv_block5.bn2.bias', 'conv_block5.bn2.running_mean', 'conv_block5.bn2.running_var','conv_block6.bn1.weight', 'conv_block6.bn1.bias', 'conv_block6.bn1.running_mean', 'conv_block6.bn1.running_var','conv_block6.bn2.weight', 'conv_block6.bn2.bias', 'conv_block6.bn2.running_mean', 'conv_block6.bn2.running_var']
        # prev_conv_key_list =  ['conv_block1.conv1.weight', 'conv_block1.conv2.weight','conv_block2.conv1.weight', 'conv_block2.conv2.weight','conv_block3.conv1.weight', 'conv_block3.conv2.weight','conv_block4.conv1.weight', 'conv_block4.conv2.weight','conv_block5.conv1.weight', 'conv_block5.conv2.weight','conv_block6.conv1.weight']


        conv_key_list = ['conv_block1.conv1.r_weight', 'conv_block1.conv1.i_weight', 'conv_block1.conv1.j_weight', 'conv_block1.conv1.k_weight',
                         'conv_block1.conv2.r_weight', 'conv_block1.conv2.i_weight', 'conv_block1.conv2.j_weight', 'conv_block1.conv2.k_weight',
                         'conv_block2.conv1.r_weight', 'conv_block2.conv1.i_weight', 'conv_block2.conv1.j_weight', 'conv_block2.conv1.k_weight',
                         'conv_block2.conv2.r_weight', 'conv_block2.conv2.i_weight', 'conv_block2.conv2.j_weight', 'conv_block2.conv2.k_weight',
                         'conv_block3.conv1.r_weight', 'conv_block3.conv1.i_weight', 'conv_block3.conv1.j_weight', 'conv_block3.conv1.k_weight',
                         'conv_block3.conv2.r_weight', 'conv_block3.conv2.i_weight', 'conv_block3.conv2.j_weight', 'conv_block3.conv2.k_weight',
                         'conv_block4.conv1.r_weight', 'conv_block4.conv1.i_weight', 'conv_block4.conv1.j_weight', 'conv_block4.conv1.k_weight',
                         'conv_block4.conv2.r_weight', 'conv_block4.conv2.i_weight', 'conv_block4.conv2.j_weight', 'conv_block4.conv2.k_weight',
                         'conv_block5.conv1.r_weight', 'conv_block5.conv1.i_weight', 'conv_block5.conv1.j_weight', 'conv_block5.conv1.k_weight',
                         'conv_block5.conv2.r_weight', 'conv_block5.conv2.i_weight', 'conv_block5.conv2.j_weight', 'conv_block5.conv2.k_weight',
                         'conv_block6.conv1.r_weight', 'conv_block6.conv1.i_weight', 'conv_block6.conv1.j_weight', 'conv_block6.conv1.k_weight',
                         'conv_block6.conv2.r_weight', 'conv_block6.conv2.i_weight', 'conv_block6.conv2.j_weight', 'conv_block6.conv2.k_weight']

        prev_conv_key_list = ['conv_block1.conv1.r_weight', 'conv_block1.conv1.i_weight', 'conv_block1.conv1.j_weight', 'conv_block1.conv1.k_weight',
                              'conv_block1.conv2.r_weight', 'conv_block1.conv2.i_weight', 'conv_block1.conv2.j_weight', 'conv_block1.conv2.k_weight',
                              'conv_block2.conv1.r_weight', 'conv_block2.conv1.i_weight', 'conv_block2.conv1.j_weight', 'conv_block2.conv1.k_weight',
                              'conv_block2.conv2.r_weight', 'conv_block2.conv2.i_weight', 'conv_block2.conv2.j_weight', 'conv_block2.conv2.k_weight',
                              'conv_block3.conv1.r_weight', 'conv_block3.conv1.i_weight', 'conv_block3.conv1.j_weight', 'conv_block3.conv1.k_weight',
                              'conv_block3.conv2.r_weight', 'conv_block3.conv2.i_weight', 'conv_block3.conv2.j_weight', 'conv_block3.conv2.k_weight',
                              'conv_block4.conv1.r_weight', 'conv_block4.conv1.i_weight', 'conv_block4.conv1.j_weight', 'conv_block4.conv1.k_weight',
                              'conv_block4.conv2.r_weight', 'conv_block4.conv2.i_weight', 'conv_block4.conv2.j_weight', 'conv_block4.conv2.k_weight',
                              'conv_block5.conv1.r_weight', 'conv_block5.conv1.i_weight', 'conv_block5.conv1.j_weight', 'conv_block5.conv1.k_weight',
                              'conv_block5.conv2.r_weight', 'conv_block5.conv2.i_weight', 'conv_block5.conv2.j_weight', 'conv_block5.conv2.k_weight',
                              'conv_block6.conv1.r_weight', 'conv_block6.conv1.i_weight', 'conv_block6.conv1.j_weight', 'conv_block6.conv1.k_weight',
                              'conv_block6.conv2.r_weight', 'conv_block6.conv2.i_weight', 'conv_block6.conv2.j_weight', 'conv_block6.conv2.k_weight']


        Z = OrderedDict()
        j = 0
        i = 0
        bn_i =0
        for key in conv_key_list:
            W_2D = weights[key].numpy()
            # print("i, key, W_2d,conv_index,weights_pruned {}".format(i), key, W_2D.shape,conv_index[key].shape ,weights_pruned[key].shape)
            if i <= 3:
                weights_pruned[key] = torch.tensor(W_2D[conv_index[key],:,:,:])
            else:
                weights_pruned[key] = torch.tensor(W_2D[conv_index[key],:,:,:][:,conv_index[prev_conv_key_list[i-4]],:,:])

            if i%4 == 0:
                # print(key,bn_key_list[j])
                # print(conv_key_list[i],conv_key_list[i+1],conv_key_list[i+2],conv_key_list[i+3]) 

                # print(weights[bn_key_list[j]].shape)
                # print(conv_index[key])
                # print(weights[bn_key_list[j]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+1]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+2]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+3]][conv_index[key]].shape)
                weights_pruned[bn_key_list[j]] = torch.cat((weights[bn_key_list[j]][conv_index[conv_key_list[i] ]], weights[bn_key_list[j]][conv_index[conv_key_list[i+1]]],
                                                            weights[bn_key_list[j]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j]][conv_index[conv_key_list[i+3]]]),dim=0) #weights[bn_key_list[j]][conv_index[key]]

                weights_pruned[bn_key_list[j+1]] =torch.cat((weights[bn_key_list[j+1]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+1]][conv_index[conv_key_list[i+1]]],
                                                             weights[bn_key_list[j+1]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+1]][conv_index[conv_key_list[i+3]]]), dim =0)

                weights_pruned[bn_key_list[j+2]] =torch.cat((weights[bn_key_list[j+2]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+2]][conv_index[conv_key_list[i+1]]],
                                                             weights[bn_key_list[j+2]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+2]][conv_index[conv_key_list[i+3]]]), dim=0)

                weights_pruned[bn_key_list[j+3]] = torch.cat((weights[bn_key_list[j+3]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+3]][conv_index[conv_key_list[i+1]]],
                                                              weights[bn_key_list[j+3]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+3]][conv_index[conv_key_list[i+3]]]), dim=0)

                j = j + 4



            i = i + 1
            # print(key)
            # weights[key] = torch.tensor(W_2D[0:50,:,:,:])
            filename = path + key + '.npy'
            # print("i, key, conv_index,weights_pruned {}".format(i), key, W_2D.shape,conv_index[key].shape ,weights_pruned[key].shape)
            # print(filename)
            # np.save(filename,sorted_index)
            # print(len(sorted_index))


        weights_pruned['fc1.r_weight'] = weights['fc1.r_weight'][conv_index['conv_block6.conv2.r_weight'],:]
        weights_pruned['fc1.i_weight'] = weights['fc1.i_weight'][conv_index['conv_block6.conv2.i_weight'],:]
        weights_pruned['fc1.j_weight'] = weights['fc1.j_weight'][conv_index['conv_block6.conv2.j_weight'],:]
        weights_pruned['fc1.k_weight'] = weights['fc1.k_weight'][conv_index['conv_block6.conv2.k_weight'],:]

        self.load_state_dict(weights_pruned)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
    
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
    
    
        x_first = torchaudio.functional.compute_deltas(x)
        x_second = torchaudio.functional.compute_deltas(x_first)
        x_third = torchaudio.functional.compute_deltas(x_second)
        #quternionic converter
        x_quaternion = torch.cat([x,x_first, x_second, x_third], dim=1) # (batch_size, 4, time_steps, freq_bins)
        #print("spectrogram shape", x_quaternion.shape)
        x = self.logmel_extractor(x_quaternion)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
    
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
    
        if self.training:
            x = self.spec_augmenter(x)
    
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
    
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
    
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        # clipwise_output = torch.log_softmax(self.fc_audioset(x))        
        # clipwise_output = nn.functional.softmax(self.fc_audioset(x))        
    
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
    
        return output_dict










        



#%% Quaternion resnet model

def _Qresnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return QuaternionConv2dNoBias(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, dilatation=1,bias=False)

def _Qresnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return QuaternionConv2dNoBias(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

class _QResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_QResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _Qresnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _Qresnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        # init_layer(self.conv1)
        init_bn(self.bn1)
        # init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
    


class _QResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_QResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _Qresnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                # init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _Qresnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                # init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
 



def after_pruning_parameters(unpruned_weights, sorted_index_current, sorted_index_previous, pruning_ratio, layer_name, prev_layer_pruned):
    Q = unpruned_weights[layer_name].numpy()
    if  prev_layer_pruned == 'yes': 
        Q1 = Q[sorted_index_current[0:int(len(sorted_index_current)*(1-pruning_ratio))],:,:,:][:,sorted_index_previous[0:int(len(sorted_index_previous)*(1-pruning_ratio))],:,:]
    else: 
        Q1 = Q[sorted_index_current[0:int(len(sorted_index_current)*(1-pruning_ratio))],:,:,:]

    after_pruning_tensor = torch.tensor(Q1)    
    
    return after_pruning_tensor


def after_pruning_bn_parameters(unpruned_weights, sorted_index, pruning_ratio, layer_name):
    Q = unpruned_weights[layer_name].numpy()
    len_Q = len(Q)
    # generate important parameters corresponding to each i,j, k, r plane
    Q1 = Q[0:int(len_Q/4)][sorted_index[0:int(len(sorted_index)*(1-pruning_ratio))]]
    Q2 = Q[int(len_Q/4):int(len_Q/2)][sorted_index[0:int(len(sorted_index)*(1-pruning_ratio))]]
    Q3 = Q[int(len_Q/2):int(len_Q*3/4)][sorted_index[0:int(len(sorted_index)*(1-pruning_ratio))]]
    Q4 = Q[int(len_Q*3/4):][sorted_index[0:int(len(sorted_index)*(1-pruning_ratio))]]
    Q_all =  np.hstack((Q1,Q2,Q3,Q4))
    
    after_pruning_tensor = torch.tensor(Q_all)    
    
    return after_pruning_tensor

def fully_connected_layers_pruning(unpruned_weights, sorted_index, pruning_ratio, layer_name):
    Q = unpruned_weights[layer_name].numpy()
    Q1 = Q[sorted_index[0:int(len(sorted_index)*(1-pruning_ratio))],:]
    after_pruning_tensor = torch.tensor(Q1)    
    
    return after_pruning_tensor

################################### QResNet38 #################################################

class  QResNet38(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(QResNet38, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = QConvBlock(in_channels=4, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = QConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = QuaternionLinear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        
        checkpoint_path_qresnet = './920000_iterations.pth'#"/Users/aryanchaudhary/surrey/audio_tagging_quaternion/pytorch/weight/Qcnn14.pth"
        checkpoint = torch.load(checkpoint_path_qresnet, map_location=torch.device('cpu')) #model.state_dict()#

        weights_teacher = checkpoint['model'] #['model']        

        self.load_state_dict(weights_teacher)


#        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        # init_layer(self.fc1)
        init_layer(self.fc_audioset)


    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x_first = torchaudio.functional.compute_deltas(x)
        x_second = torchaudio.functional.compute_deltas(x_first)
        x_third = torchaudio.functional.compute_deltas(x_second) 
        #quternionic converter
        x_quaternion = torch.cat([x,x_first, x_second, x_third], dim=1) # (batch_size, 4, time_steps, freq_bins)
        #print("spectrogram shape", x_quaternion.shape)
        x = self.logmel_extractor(x_quaternion)    # (batch_size, 4, time_steps, mel_bins)
        #print("logmel shape", x.shape)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


################################# QResnet pruned models #########################################

class QResNet38_pruned(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(QResNet38_pruned, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        p = 0.25   # pruning ratio

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = QConvBlock(in_channels=4, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _QResNet(block=_QResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = QConvBlock(in_channels=512, out_channels=int(2048*(1-p)))

        self.fc1 = QuaternionLinear(int(2048*(1-p)), 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        # self.init_weights()
        # pruning pipleine..............................
        # user input
        checkpoint_path_qresnet = './920000_iterations.pth'#"/Users/aryanchaudhary/surrey/audio_tagging_quaternion/pytorch/weight/Qcnn14.pth"
        checkpoint = torch.load(checkpoint_path_qresnet, map_location=torch.device('cpu')) #model.state_dict()#

        weights = checkpoint['model'] #['model']
        # weights_pruned = weights 

        w_Qresnet_unpruned  = weights

        w_Qresnet_pruned = weights
        # user input 
        path_to_qresent_sorted_index = 'as' #add path here.........................................................................................................
        conv_block_after1_conv1_important_index = np.load(os.path.join(path_to_qresent_sorted_index,'conv_block_after1.conv1_mean_score.npy'))
        conv_block_after1_conv2_important_index = np.load(os.path.join(path_to_qresent_sorted_index,'conv_block_after1.conv2_mean_score.npy'))

        sorted_index_bottleneck = np.arange(0,int(512/4))

        sorted_index_conv1 =  conv_block_after1_conv1_important_index
        sorted_index_conv2 =  conv_block_after1_conv2_important_index
        w_Qresnet_pruned['conv_block_after1.conv1.i_weight'] =  after_pruning_parameters(w_Qresnet_unpruned, sorted_index_conv1, sorted_index_bottleneck, pruning_ratio = p, layer_name = 'conv_block_after1.conv1.i_weight', prev_layer_pruned = 'no') # (out, input, filter size)

        w_Qresnet_pruned['conv_block_after1.conv1.j_weight'] =  after_pruning_parameters(w_Qresnet_unpruned, sorted_index_conv1, sorted_index_bottleneck, pruning_ratio = p, layer_name = 'conv_block_after1.conv1.j_weight', prev_layer_pruned = 'no') # (out, input, filter size)

        w_Qresnet_pruned['conv_block_after1.conv1.k_weight'] =  after_pruning_parameters(w_Qresnet_unpruned, sorted_index_conv1, sorted_index_bottleneck, pruning_ratio = p, layer_name = 'conv_block_after1.conv1.k_weight', prev_layer_pruned = 'no') # (out, input, filter size)

        w_Qresnet_pruned['conv_block_after1.conv1.r_weight'] =  after_pruning_parameters(w_Qresnet_unpruned, sorted_index_conv1, sorted_index_bottleneck, pruning_ratio = p, layer_name = 'conv_block_after1.conv1.r_weight', prev_layer_pruned = 'no') # (out, input, filter size)

        w_Qresnet_pruned['conv_block_after1.bn1.running_mean'] = after_pruning_bn_parameters(w_Qresnet_unpruned, conv_block_after1_conv1_important_index, pruning_ratio = p, layer_name = 'conv_block_after1.bn1.running_mean')

        w_Qresnet_pruned['conv_block_after1.bn1.running_var'] = after_pruning_bn_parameters(w_Qresnet_unpruned, conv_block_after1_conv1_important_index, pruning_ratio = p, layer_name = 'conv_block_after1.bn1.running_var')

        w_Qresnet_pruned['conv_block_after1.bn1.bias'] = after_pruning_bn_parameters(w_Qresnet_unpruned, conv_block_after1_conv1_important_index, pruning_ratio = p, layer_name = 'conv_block_after1.bn1.bias')

        w_Qresnet_pruned['conv_block_after1.bn1.weight'] = after_pruning_bn_parameters(w_Qresnet_unpruned, conv_block_after1_conv1_important_index, pruning_ratio = p, layer_name = 'conv_block_after1.bn1.weight')

        w_Qresnet_pruned['conv_block_after1.conv2.i_weight'] =  after_pruning_parameters(w_Qresnet_unpruned, sorted_index_conv2, sorted_index_conv1, pruning_ratio = p, layer_name = 'conv_block_after1.conv2.i_weight',prev_layer_pruned = 'yes') # (out, input, filter size)

        w_Qresnet_pruned['conv_block_after1.conv2.j_weight'] =  after_pruning_parameters(w_Qresnet_unpruned,  sorted_index_conv2, sorted_index_conv1, pruning_ratio = p, layer_name = 'conv_block_after1.conv2.j_weight', prev_layer_pruned = 'yes') # (out, input, filter size)

        w_Qresnet_pruned['conv_block_after1.conv2.k_weight'] =  after_pruning_parameters(w_Qresnet_unpruned,  sorted_index_conv2, sorted_index_conv1, pruning_ratio = p, layer_name = 'conv_block_after1.conv2.k_weight', prev_layer_pruned = 'yes') # (out, input, filter size)

        w_Qresnet_pruned['conv_block_after1.conv2.r_weight'] =  after_pruning_parameters(w_Qresnet_unpruned,  sorted_index_conv2, sorted_index_conv1, pruning_ratio = p, layer_name = 'conv_block_after1.conv2.r_weight', prev_layer_pruned = 'yes') # (out, input, filter size)

        w_Qresnet_pruned['conv_block_after1.bn2.running_mean'] = after_pruning_bn_parameters(w_Qresnet_unpruned, sorted_index_conv2, pruning_ratio = p, layer_name = 'conv_block_after1.bn2.running_mean')

        w_Qresnet_pruned['conv_block_after1.bn2.running_var'] = after_pruning_bn_parameters(w_Qresnet_unpruned, sorted_index_conv2, pruning_ratio = p, layer_name = 'conv_block_after1.bn2.running_var')

        w_Qresnet_pruned['conv_block_after1.bn2.bias'] = after_pruning_bn_parameters(w_Qresnet_unpruned, sorted_index_conv2, pruning_ratio = p, layer_name = 'conv_block_after1.bn2.bias')

        w_Qresnet_pruned['conv_block_after1.bn2.weight'] = after_pruning_bn_parameters(w_Qresnet_unpruned, sorted_index_conv2, pruning_ratio = p, layer_name = 'conv_block_after1.bn2.weight')


        w_Qresnet_pruned['fc1.i_weight']   =  fully_connected_layers_pruning(w_Qresnet_unpruned, sorted_index_conv2, pruning_ratio = p, layer_name = 'fc1.i_weight')
        w_Qresnet_pruned['fc1.j_weight']   =  fully_connected_layers_pruning(w_Qresnet_unpruned, sorted_index_conv2, pruning_ratio = p, layer_name = 'fc1.j_weight')
        w_Qresnet_pruned['fc1.k_weight']   =  fully_connected_layers_pruning(w_Qresnet_unpruned, sorted_index_conv2, pruning_ratio = p, layer_name = 'fc1.k_weight')
        w_Qresnet_pruned['fc1.r_weight']   =  fully_connected_layers_pruning(w_Qresnet_unpruned, sorted_index_conv2, pruning_ratio = p, layer_name = 'fc1.r_weight')

        self.load_state_dict(w_Qresnet_pruned)
        
        
    def init_weights(self):
        init_bn(self.bn0)
        # init_layer(self.fc1)
        init_layer(self.fc_audioset)


    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x_first = torchaudio.functional.compute_deltas(x)
        x_second = torchaudio.functional.compute_deltas(x_first)
        x_third = torchaudio.functional.compute_deltas(x_second) 
        #quternionic converter
        x_quaternion = torch.cat([x,x_first, x_second, x_third], dim=1) # (batch_size, 4, time_steps, freq_bins)
        #print("spectrogram shape", x_quaternion.shape)
        x = self.logmel_extractor(x_quaternion)    # (batch_size, 4, time_steps, mel_bins)
        #print("logmel shape", x.shape)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict












    

