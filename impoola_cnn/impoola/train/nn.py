import numpy as np
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
    elif encoder_type == 'impala_new':
        micro_p = kwargs.pop('micro_dropout_p', 0.03)
        gauss_std = kwargs.pop('gauss_std', 1e-3)
        force_head_silu = kwargs.pop('force_head_silu', True)
        base_model = ImpalaCNN(*args, **kwargs)
        model = ImpalaCNNWithTinyReg(base_model, micro_dropout_p=micro_p, gauss_std=gauss_std,
                                     force_head_silu=force_head_silu)
        out_features = kwargs['out_features']
        return model, out_features
    else:
        raise NotImplementedError


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, scale, use_layer_init_normed=False, activation='relu'):
        super().__init__()
        # scale = (1/3**0.5 * 1/2**0.5)**0.5 # For default IMPALA CNN this is the final scale value in the PPG code
        # scale = np.sqrt(scale)
        kernel_size = 3
        conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        # self.conv0 = layer_init_kaiming_uniform(conv0)
        self.conv0 = conv0
        # self.conv0 = layer_init_normed(conv0, norm_dim=(1, 2, 3), scale=scale) if use_layer_init_normed else conv0

        conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        # self.conv1 = layer_init_normed(conv1, norm_dim=(1, 2, 3), scale=scale) if use_layer_init_normed else conv1
        # self.conv1 = layer_init_kaiming_uniform(conv1)
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

        # Input convolution and pooling
        conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                         padding="same")
        # self.conv = layer_init_normed(conv, norm_dim=(1, 2, 3), scale=1.0) if use_layer_init_normed else conv
        # self.conv = layer_init_kaiming_uniform(conv) #, nonlinearity="linear")
        self.conv = conv

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
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
        # assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpalaCNN(nn.Module):
    def __init__(
            self, envs,
            width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu',
            use_layer_init_normed=False,

    ):
        super().__init__()

        shape = envs.single_observation_space.shape  # (c, h, w)
        scale = 1 / np.sqrt(len(cnn_filters))  # Not fully sure about the logic behind this but it's used in PPG code

        # CNN backbone
        cnn_layers = []

        for i_block, out_channels in enumerate(cnn_filters):
            conv_seq = ConvSequence(shape, int(out_channels * width_scale), scale=scale,
                                    use_layer_init_normed=use_layer_init_normed, activation=activation)

            shape = conv_seq.get_output_shape()
            cnn_layers.append(conv_seq)

        # TODO: Breaking change to before (Jan 24)
        cnn_layers += [activation_factory(activation)]

        # ImpoolaCNN improves the original IMPALA CNN by adding a pooling layer
        cnn_layers += [nn.AdaptiveAvgPool2d((1, 1))]

        # Linear head
        linear_layers = cnn_layers
        linear_layers += [nn.Flatten()]

        # encodertop = nn.LazyLinear(out_features)  # in_features=shape[0] * shape[1] * shape[2]

        in_features_encoder = shape[0] * 1 * 1  # after adaptive avg pooling to (1,1)

        encodertop = nn.Linear(in_features_encoder, out_features=out_features)

        # encodertop = layer_init_kaiming_uniform(encodertop)  # TODO: Orthogonal could be better

        # encodertop = nn.LazyLinear(out_features * 2)  # in_features=shape[0] * shape[1] * shape[2]
        # encodertop = layer_init_normed(encodertop, norm_dim=1, scale=1.4) if use_layer_init_normed else encodertop

        linear_layers += [
            # activation_factory(activation),
            encodertop,
            activation_factory(activation)
        ]
        self.network = nn.Sequential(*linear_layers)

    def forward(self, x):
        # add a positional signal as overlay to the input image
        # x = x + self.positional_signal(x)
        x = x / 255.0
        return self.network(x)

    def get_output_shape(self):
        return self.network[-2].out_features


import numpy as np
import torch
import torch.nn as nn


class TrainOnlyMicroDropoutTiny(nn.Module):
    def __init__(self, p=0.03):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        if self.training:
            return self.dropout(x)
        return x


class TrainOnlyGaussianNoiseTiny(nn.Module):
    def __init__(self, std=1e-3):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class HeadActivationSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x)


class EarlyActivationReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x)


class ImpalaCNNWithTinyReg(nn.Module):
    def __init__(self, base_encoder: nn.Module, micro_dropout_p=0.03, gauss_std=1e-3, force_head_silu=True):
        super().__init__()
        self.base = base_encoder
        self.after_pool_dropout = TrainOnlyMicroDropoutTiny(p=micro_dropout_p)
        self.after_pool_noise = TrainOnlyGaussianNoiseTiny(std=gauss_std)
        self.after_linear_dropout = TrainOnlyMicroDropoutTiny(p=micro_dropout_p)
        self.after_linear_noise = TrainOnlyGaussianNoiseTiny(std=gauss_std)
        self.force_head_silu = force_head_silu
        self._wire()

    def _wire(self):
        if not hasattr(self.base, "network"):
            raise AttributeError("base_encoder is expected to have attribute .network (nn.Sequential)")
        net = self.base.network
        if not isinstance(net, nn.Sequential):
            raise TypeError("base_encoder.network must be nn.Sequential")
        pool_idx = None
        for i, m in enumerate(net):
            if isinstance(m, nn.AdaptiveAvgPool2d):
                pool_idx = i
                break
        if pool_idx is None:
            raise RuntimeError("Could not find AdaptiveAvgPool2d in base_encoder.network")
        modules = list(net)
        insert_pos = pool_idx + 1
        modules.insert(insert_pos, self.after_pool_dropout)
        modules.insert(insert_pos + 1, self.after_pool_noise)
        last_linear_idx = None
        for i, m in enumerate(modules):
            if isinstance(m, nn.Linear):
                last_linear_idx = i
        if last_linear_idx is None:
            raise RuntimeError("Could not find Linear head in base_encoder.network")
        insert_pos2 = last_linear_idx + 1
        modules.insert(insert_pos2, self.after_linear_dropout)
        modules.insert(insert_pos2 + 1, self.after_linear_noise)
        if self.force_head_silu:
            if insert_pos2 + 2 < len(modules):
                modules[insert_pos2 + 2] = HeadActivationSiLU()
        self.base.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.base(x)

    def get_output_shape(self):
        if hasattr(self.base, "get_output_shape"):
            return self.base.get_output_shape()
        if isinstance(self.base.network[-2], nn.Linear):
            return self.base.network[-2].out_features
        raise AttributeError("Cannot infer output shape from base encoder")
