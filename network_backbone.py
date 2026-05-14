import os
import numpy as np
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F

from MinkowskiEngine import CoordinateMapKey
from MinkowskiEngineBackend._C import CoordinateMapKey, ConvolutionMode


from config_load import get_config
args = get_config().parse_args()
CONV_TYPE = args.conv_type

#####################  Basic Operations  ############################
class NetworkFactory:
    @staticmethod
    def create_SparseConv3d(implementation=CONV_TYPE, *args, **kwargs):
        implementations = {
            '': ME.MinkowskiConvolution,
            'ME': ME.MinkowskiConvolution,
            'quant': QuantMinkowskiConvolution,
            }
        
        return implementations[implementation](*args, **kwargs)

    @staticmethod
    def create_SparseTransposeConv3d(implementation=CONV_TYPE, *args, **kwargs):
        implementations = {
            '': ME.MinkowskiConvolutionTranspose,
            'ME': ME.MinkowskiConvolutionTranspose,
            'quant': QuantMinkowskiConvolutionTranspose,
            }
        
        return implementations[implementation](*args, **kwargs)

    @staticmethod
    def create_SparseLinear(implementation=CONV_TYPE, *args, **kwargs):
        implementations = {
            '': ME.MinkowskiLinear,
            'ME': ME.MinkowskiLinear,
            'quant': QuantMinkowskiLinear,
            }
        
        return implementations[implementation](*args, **kwargs)

    @staticmethod
    def create_ActivationLayer(implementation='relu', *args, **kwargs):
        implementations = {
            'relu': ME.MinkowskiReLU,
            'gelu': MinkowskiGELU,
            'prelu': ME.MinkowskiPReLU,
            '': torch.nn.Identity,
            }
        
        return implementations[implementation](*args, **kwargs)


#################################### quantized Layers ####################################

def differentiable_quantizer(x, bits=8, method='per_tensor_symetric', return_quant=True):
    if method=='per_tensor_symetric':
        # -127, 127
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax
        # scale & zero_point
        x_min, x_max = torch.aminmax(x)
        scale = torch.max(torch.abs(x_min), torch.abs(x_max)) / qmax
        zero_point = torch.tensor(0.0, device=x.device)
    if method=='per_tensor_affine':
        # 0, 255
        qmin = 0
        qmax = 2 ** bits - 1
        # scale & zero_point
        # x_min, x_max = torch.aminmax(x)
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / qmax
        zero_point = x_min

    # linear quantization
    x_scaled = (x-zero_point) / scale
    x_clamped = torch.clamp(x_scaled, qmin, qmax)
    x_rounded = torch.round(x_clamped)
    x_quantized = x_rounded * scale + zero_point
    
    # fake quantize
    q_x = x + (x_quantized - x).detach()
    
    if return_quant==False:
        return q_x
    else:
        return q_x, (q_x-zero_point) / scale

##################################################################################

class QuantMinkowskiLinear(ME.MinkowskiLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
    ):
        super().__init__(
       in_features,
        out_features,
        bias=True,
        )

        self.int_weight = None
        self.int_bias = None

    def forward(self, input):

        quantized_weight, int_weight = differentiable_quantizer(self.linear.weight)
        quantized_bias, int_bias = differentiable_quantizer(self.linear.bias)

        self.int_weight = int_weight
        self.int_bias = int_bias

        outfeat = F.linear(input.F, quantized_weight, quantized_bias)

        return ME.SparseTensor(features=outfeat, 
                            coordinate_map_key=input.coordinate_map_key,
                            coordinate_manager=input.coordinate_manager, 
                            device=input.device)
        

##################################################################################

class QuantMinkowskiConvolution(ME.MinkowskiConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=True,
        dimension=3,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            dimension=dimension
        )
        
        self.int_kernel = None
        self.int_bias = None
    
    def forward(self, input):
        quantized_kernel, int_kernel = differentiable_quantizer(self.kernel)
        quantized_bias, int_bias = differentiable_quantizer(self.bias)

        self.int_kernel = int_kernel
        self.int_bias = int_bias

        out_coordinate_map_key = CoordinateMapKey(
                input.coordinate_map_key.get_coordinate_size()
            )

        outfeat = ME.MinkowskiConvolutionFunction.apply(input.F,
                                                        quantized_kernel,
                                                        self.kernel_generator,
                                                        ConvolutionMode.DEFAULT,
                                                        input.coordinate_map_key,
                                                        out_coordinate_map_key,
                                                        input._manager)
        outfeat += quantized_bias

        return ME.SparseTensor(outfeat,
            coordinate_map_key=out_coordinate_map_key,
            coordinate_manager=input._manager)
    

##################################################################################

class QuantMinkowskiConvolutionTranspose(ME.MinkowskiConvolutionTranspose):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=True,
        dimension=3,
        expand_coordinates=True):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            dimension=dimension,
            expand_coordinates=expand_coordinates)
        
        self.int_kernel = None
        self.int_bias = None
        
    def forward(self, input):
        # quantized_weight = self.kernel
        # quantized_bias = self.bias

        quantized_kernel, int_kernel = differentiable_quantizer(self.kernel)
        quantized_bias, int_bias = differentiable_quantizer(self.bias)

        self.int_kernel = int_kernel
        self.int_bias = int_bias

        out_coordinate_map_key = CoordinateMapKey(
                input.coordinate_map_key.get_coordinate_size()
            )

        outfeat = ME.MinkowskiConvolutionTransposeFunction.apply(input.F,
                                                        quantized_kernel,
                                                        self.kernel_generator,
                                                        ConvolutionMode.DEFAULT,
                                                        input.coordinate_map_key,
                                                        out_coordinate_map_key,
                                                        input._manager)
        outfeat += quantized_bias

        return ME.SparseTensor(outfeat,
            coordinate_map_key=out_coordinate_map_key,
            coordinate_manager=input._manager)
    


################################################
class MinkowskiGELU(torch.nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
        
    def forward(self, inputs):
        output = ME.SparseTensor(features=self.gelu(inputs.F),  
                                coordinate_manager=inputs.coordinate_manager,  
                                coordinate_map_key=inputs.coordinate_map_key,
                                device=inputs.device)
        
        return output
    

######################### ResNet #########################
class ResNet(torch.nn.Module): 
    """Residual Network
    """  
    def __init__(self, channels, kernel_size=3, dimension=3):
        super().__init__()
        self.conv0 = NetworkFactory.create_SparseConv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        self.conv1 = NetworkFactory.create_SparseConv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=dimension)
        self.relu = NetworkFactory.create_ActivationLayer()


    def forward(self, x):
        out = self.conv0(x)
        out = self.relu(out)
        out = self.conv1(out)
        out += x

        return out

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels=1, channels=32, out_channels=1, 
                kernel_size=3, block_layers=3, dimension=3,
                global_residual=False):
        super().__init__()
        self.global_residual = global_residual
        self.layer_in = NetworkFactory.create_SparseLinear(in_features=in_channels, out_features=channels)
        self.out_layer = NetworkFactory.create_SparseLinear(in_features=channels, out_features=out_channels)
        
        self.layers = torch.nn.ModuleList()
        for i in range(block_layers):
            self.layers.append(ResNet(channels=channels, kernel_size=kernel_size, 
                                      dimension=dimension))

    def forward(self, x):
        out0 = self.layer_in(x)
        out = out0
        for resnet in self.layers:
            out = resnet(out)
        if len(self.layers)>1 and self.global_residual:
            out += out0
        out = self.out_layer(out)
        
        return out

class LinearLayers(torch.nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()

        self.layers = torch.nn.Sequential(
            NetworkFactory.create_SparseLinear(in_features=in_channels, out_features=channels),
            NetworkFactory.create_ActivationLayer(),
            NetworkFactory.create_SparseLinear(in_features=channels, out_features=channels),
            NetworkFactory.create_ActivationLayer(),
            NetworkFactory.create_SparseLinear(in_features=channels, out_features=out_channels))

    
    def forward(self, x):
        out = self.layers(x)

        return out


