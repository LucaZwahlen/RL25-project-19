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
    if encoder_type == 'new_version':
        model = NewVersion(*args, **kwargs)
        out_features = kwargs['out_features']
        return model, out_features
    raise NotImplementedError


class ResidualBlock(nn.Module):
    def __init__(self, channels, scale, use_layer_init_normed=False, activation='relu'):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding='same')
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding='same')
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
    def __init__(self, input_shape, out_channels, scale, use_layer_init_normed=False, activation='relu'):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding="same")
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        nblocks = 2
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale / np.sqrt(nblocks), use_layer_init_normed=use_layer_init_normed, activation=activation)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale / np.sqrt(nblocks), use_layer_init_normed=use_layer_init_normed, activation=activation)

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
    def __init__(self, envs, width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu', use_layer_init_normed=False):
        super().__init__()
        shape = envs.single_observation_space.shape
        scale = 1 / np.sqrt(len(cnn_filters))
        cnn_layers = []
        for out_channels in cnn_filters:
            conv_seq = ConvSequence(shape, int(out_channels * width_scale), scale=scale, use_layer_init_normed=use_layer_init_normed, activation=activation)
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


class AddCoords(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xx, yy], dim=1)


class MHAAggregator(nn.Module):
    def __init__(self, channels, out_features, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, out_features)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        y, _ = self.mha(tokens, tokens, tokens)
        y = self.norm(y)
        y = y.mean(dim=1)
        return self.proj(y)


class NewVersion(nn.Module):
    def __init__(self, envs, width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu', use_layer_init_normed=False, num_heads=4):
        super().__init__()
        obs_c, obs_h, obs_w = envs.single_observation_space.shape
        self.add_coords = AddCoords()
        in_channels = obs_c + 2
        scale = 1 / np.sqrt(len(cnn_filters))
        cnn_layers = []
        shape = (in_channels, obs_h, obs_w)
        for out_channels in cnn_filters:
            conv_seq = ConvSequence(shape, int(out_channels * width_scale), scale=scale, use_layer_init_normed=use_layer_init_normed, activation=activation)
            shape = conv_seq.get_output_shape()
            cnn_layers.append(conv_seq)
        cnn_layers += [activation_factory(activation)]
        self.backbone = nn.Sequential(*cnn_layers)
        self.aggregator = MHAAggregator(shape[0], out_features, num_heads=num_heads)

    def forward(self, x):
        x = x / 255.0
        x = self.add_coords(x)
        x = self.backbone(x)
        z = self.aggregator(x)
        return z

    def get_output_shape(self):
        return self.aggregator.proj.out_features
