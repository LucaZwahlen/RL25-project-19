import numpy as np
import torch
import torch.nn as nn


def layer_init_orthogonal(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def activation_factory(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif activation == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation == 'gelu':
        return nn.GELU(approximate='tanh')
    elif activation == 'silu':
        return nn.SiLU(inplace=True)
    else:
        raise NotImplementedError



def encoder_factory(encoder_type, *args, **kwargs):
    if encoder_type == 'impala':
        model = ImpalaCNN(*args, **kwargs)
        out_features = kwargs['out_features']
        return model, out_features
    elif encoder_type == 'impala_new':
        out_features = kwargs['out_features']
        model = ImpalaCnnReg(*args, **kwargs)
        return model, out_features
    else:
        raise NotImplementedError


class ResidualBlock(nn.Module):
    def __init__(self, channels, scale, use_layer_init_normed=False, activation='relu'):
        super().__init__()
        kernel_size = 3
        conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        self.conv0 = conv0
        conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        self.conv1 = conv1
        self.activation0 = activation_factory(activation)
        self.activation1 = activation_factory(activation)

    def forward(self, x):
        inputs = x
        x = self.activation0(x)
        x = self.conv0(x)
        x = self.activation1(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, scale, use_layer_init_normed=False, activation='relu', ):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels

        conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                         padding="same")
        self.conv = conv
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        nblocks = 2
        scale = scale / np.sqrt(nblocks)
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale, use_layer_init_normed=use_layer_init_normed,
                                        activation=activation)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale, use_layer_init_normed=use_layer_init_normed,
                                        activation=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpalaCNN(nn.Module):
    def __init__(
            self,
            envs,
            width_scale=1,
            out_features=256,
            cnn_filters=(16, 32, 32),
            activation='relu',
            use_layer_init_normed=False,
            p_augment=0.0,
            micro_dropout_p=0.0
    ):
        super().__init__()

        self.augment = TrainOnlyBlurNoise(p=p_augment)
        shape = envs.single_observation_space.shape  # (c, h, w)
        scale = 1 / np.sqrt(len(cnn_filters))  # Not fully sure about the logic behind this but it's used in PPG code

        cnn_layers = []
        for i_block, out_channels in enumerate(cnn_filters):
            conv_seq = ConvSequence(shape, int(out_channels * width_scale), scale=scale,
                                    use_layer_init_normed=use_layer_init_normed, activation=activation)

            shape = conv_seq.get_output_shape()
            cnn_layers.append(conv_seq)

        cnn_layers += [activation_factory(activation)]

        # ImpoolaCNN improves the original IMPALA CNN by adding a pooling layer
        cnn_layers += [nn.AdaptiveAvgPool2d((1, 1))]

        linear_layers = cnn_layers
        linear_layers += [nn.Flatten()]

        in_features_encoder = shape[0] * 1 * 1

        encodertop = nn.Linear(in_features_encoder, out_features=out_features)

        linear_layers += [
            encodertop,
            activation_factory(activation)
        ]
        self.network = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = x.float().mul_(1.0 / 255.0)
        x = self.augment(x)
        return self.network(x)

    def get_output_shape(self):
        return self.network[-2].out_features


#############################################################
# Regularized Impala CNN with train-time-only dropout and noise
#############################################################

class TrainOnlyMicroDropout(nn.Module):
    def __init__(self, p=0.03):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        if self.training:
            return self.dropout(x)
        return x


class TrainOnlyBlurNoise(nn.Module):
    def __init__(self, p=0.1, noise_std=1e-3):
        super().__init__()
        self.p = float(p)
        self.noise_std = float(noise_std)
        self.blur = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        if not self.training or self.p <= 0.0 or torch.rand(()) >= self.p:
            return x
        y = x.float()
        y = self.blur(y)
        if self.noise_std > 0.0:
            y = y + torch.randn_like(y) * self.noise_std
        return y


class EarlyActivationReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x)


class ImpalaCnnReg(nn.Module):
    def __init__(self,
                 envs,
                 width_scale=1,
                 out_features=256,
                 cnn_filters=(16, 32, 32),
                 activation='relu',
                 use_layer_init_normed=False,
                 micro_dropout_p=0.03,
                 p_augment=0.1):
        super().__init__()
        self.base = ImpalaCNN(envs=envs,
                              width_scale=width_scale,
                              out_features=out_features,
                              cnn_filters=cnn_filters,
                              activation=activation,
                              use_layer_init_normed=use_layer_init_normed,
                              p_augment=p_augment)
        self.after_pool_dropout = TrainOnlyMicroDropout(p=micro_dropout_p)
        self.after_linear_dropout = TrainOnlyMicroDropout(p=micro_dropout_p)
        self._wire()

    def _wire(self):
        net = self.base.network
        modules = list(net)
        pool_idx = None
        for i, m in enumerate(modules):
            if isinstance(m, nn.AdaptiveAvgPool2d):
                pool_idx = i
                break
        if pool_idx is None:
            self.base.network = nn.Sequential(*modules)
            return
        insert_pos = pool_idx + 1
        modules.insert(insert_pos, self.after_pool_dropout)
        last_linear_idx = None
        for i, m in enumerate(modules):
            if isinstance(m, nn.Linear):
                last_linear_idx = i
        if last_linear_idx is not None:
            insert_pos2 = last_linear_idx + 1
            modules.insert(insert_pos2, self.after_linear_dropout)
        self.base.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.base(x)

    def get_output_shape(self):
        return self.base.get_output_shape()
