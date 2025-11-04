import numpy as np
import torch
import torch.nn as nn


def layer_init_orthogonal(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def activation_factory(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'rrelu':
        return nn.RReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU()
    else:
        raise NotImplementedError


def encoder_factory(encoder_type, *args, **kwargs):
    if encoder_type == 'impala':
        model = ImpalaCNN(*args, **kwargs)
        out_features = kwargs['out_features']
        return model, out_features
    if encoder_type == 'impoola_plus':
        model = ImpoolaPlusCNN(*args, **kwargs)
        out_features = kwargs['out_features']
        return model, out_features
    raise NotImplementedError


class ResidualBlock(nn.Module):
    def __init__(self, channels, scale, use_layer_init_normed=False, activation='relu'):
        super().__init__()
        self.act0 = activation_factory(activation)
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = activation_factory(activation)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.conv0.weight, np.sqrt(2))
        nn.init.zeros_(self.conv0.bias)
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.scale = float(scale)

    def forward(self, x):
        y = self.act0(x)
        y = self.conv0(y)
        y = self.act1(y)
        y = self.conv1(y)
        return x + self.scale * y


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, scale, use_layer_init_normed=False, activation='relu',
                 pool_type='avg'):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(self._input_shape[0], self._out_channels, kernel_size=3, padding=1)
        self.pooling = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) if pool_type == 'avg' else nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)
        nblocks = 2
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale / np.sqrt(nblocks),
                                        use_layer_init_normed=use_layer_init_normed, activation=activation)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale / np.sqrt(nblocks),
                                        use_layer_init_normed=use_layer_init_normed, activation=activation)
        nn.init.orthogonal_(self.conv.weight, np.sqrt(2))
        nn.init.zeros_(self.conv.bias)

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
    def __init__(self, envs, width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu',
                 use_layer_init_normed=False):
        super().__init__()
        shape = envs.single_observation_space.shape
        scale = 1 / np.sqrt(len(cnn_filters))
        cnn_layers = []
        for out_channels in cnn_filters:
            conv_seq = ConvSequence(shape, int(out_channels * width_scale), scale=scale,
                                    use_layer_init_normed=use_layer_init_normed, activation=activation, pool_type='avg')
            shape = conv_seq.get_output_shape()
            cnn_layers.append(conv_seq)
        cnn_layers += [activation_factory(activation)]
        cnn_layers += [nn.AdaptiveAvgPool2d((1, 1))]
        linear_layers = cnn_layers
        linear_layers += [nn.Flatten()]
        in_features_encoder = shape[0]
        encodertop = nn.Linear(in_features_encoder, out_features=out_features)
        linear_layers += [encodertop, activation_factory(activation)]
        self.network = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = x / 255.0
        return self.network(x)

    def get_output_shape(self):
        return self.network[-2].out_features


class ImpoolaPlusCNN(nn.Module):
    def __init__(self, envs, width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu',
                 use_layer_init_normed=False):
        super().__init__()
        shape = envs.single_observation_space.shape
        scale = 1 / np.sqrt(len(cnn_filters))
        cnn_layers = []
        for out_channels in cnn_filters:
            conv_seq = ConvSequence(shape, int(out_channels * width_scale), scale=scale,
                                    use_layer_init_normed=use_layer_init_normed, activation=activation, pool_type='avg')
            shape = conv_seq.get_output_shape()
            cnn_layers.append(conv_seq)
        self.cnn = nn.Sequential(*cnn_layers)
        self.act = activation_factory(activation)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        in_features_encoder = shape[0] * 2
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features_encoder, out_features),
            activation_factory(activation)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        x = self.act(x)
        a = self.gap(x)
        m = self.gmp(x)
        x = torch.cat([a, m], dim=1)
        x = self.proj(x)
        return x

    def get_output_shape(self):
        return self.proj[1].out_features
