#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""U-Net model with transfer learning from the best model so far."""

# -- File info -- #
__author__ = ['Andreas R. Stokholm', 'Magnus Ostertag']
__contributor__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.3.0'
__date__ = '2022-09-20'


# -- Third-party modules -- #
import torch


class UNetTrans(torch.nn.Module):
    """PyTorch U-Net Class using transfer learning from another model."""

    def __init__(self, options):
        # initialize the new model as a 
        super().__init__()

        contract_n_oldModel = len(options['transfer_model_architecture']['unet_conv_filters']) - 1

        # check whether we can even transfer this model
        for different_option in options['transfer_model_architecture']:
            if different_option != 'unet_conv_filters':
                raise ValueError("""The transfer model architecture is not compatible with other differences than the filters not implemented, yet.""")

        if options['transfer_model_architecture']['unet_conv_filters'] != options['unet_conv_filters'][:contract_n_oldModel+1]:
            raise ValueError("""The transfer model architecture is not compatible as the filters of the old model have to be a subset of the new model.""")

        # initialize the model to transfer from
        oldModel_states = torch.load(options['transfer_model_path'])['model_state_dict']
        # print(oldModel_states.keys())

        # contract_n_newModel = len(options['unet_conv_filters']) - 1

        # stich downsampling part of the old model to the new model as its first part
        # then stich the upsampling part of the old model to the new model as its last part
        self.input_block = DoubleConv(options,
                                      input_n=len(options['train_variables']), 
                                      output_n=options['unet_conv_filters'][0], 
                                      weights_list=[oldModel_states['input_block.double_conv.0.weight'],
                                                    #  oldModel_states['input_block.double_conv.0.bias']],
                                                    oldModel_states['input_block.double_conv.3.weight']])
                                                    #  oldModel_states['input_block.double_conv.3.bias']]])

        self.contract_blocks = torch.nn.ModuleList()
        for contract_n in range(1, len(options['unet_conv_filters'])):
            if contract_n <= contract_n_oldModel:  # which is then one smaller than the number of contract blocks in the old model
                weights_list_cont = [oldModel_states[f'contract_blocks.{contract_n - 1}.double_conv.double_conv.0.weight'],
                                    #   oldModel_states[f'contract_blocks.{contract_n - 1}.double_conv.double_conv.0.bias']],
                                     oldModel_states[f'contract_blocks.{contract_n - 1}.double_conv.double_conv.3.weight'],]
                                    #   oldModel_states[f'contract_blocks.{contract_n - 1}.double_conv.double_conv.3.bias']]]
            self.contract_blocks.append(
                ContractingBlock(options=options,
                                 input_n=options['unet_conv_filters'][contract_n - 1],
                                 output_n=options['unet_conv_filters'][contract_n],
                                 weights_list=weights_list_cont))  # only used to contract input patch.

        self.bridge = ContractingBlock(options,
                                       input_n=options['unet_conv_filters'][-1],
                                       output_n=options['unet_conv_filters'][-1],
                                       weights_list=[oldModel_states['bridge.double_conv.double_conv.0.weight'], #, 'bridge.double_conv.double_conv.0.bias'],
                                                     oldModel_states['bridge.double_conv.double_conv.3.weight'],]) # 'bridge.double_conv.double_conv.3.bias']])

        weights_list = None
        old_expand_i = 0
        if len(options['transfer_model_architecture']['unet_conv_filters']) == len(options['unet_conv_filters']):
            weights_list = [oldModel_states['expand_blocks.0.double_conv.double_conv.0.weight'], # 'expand_blocks.0.double_conv.double_conv.0.bias'],
                            oldModel_states['expand_blocks.0.double_conv.double_conv.3.weight'],] # 'expand_blocks.0.double_conv.double_conv.3.bias']]
            old_expand_i = 1

        self.expand_blocks = torch.nn.ModuleList()
        self.expand_blocks.append(
            ExpandingBlock(options=options,
                           input_n=options['unet_conv_filters'][-1],
                           output_n=options['unet_conv_filters'][-1],
                           weights_list=weights_list))

        for expand_n in range(len(options['unet_conv_filters']), 1, -1):
            weights_list_exp = None
            if expand_n <= (contract_n_oldModel+2):  # will only go down to 2 ??
                # print(f'expand_n: {expand_n}, old_expand_i: {old_expand_i}, len(options[unet_conv_filters]): {len(options["unet_conv_filters"])}')
                weights_list_exp = [oldModel_states[f'expand_blocks.{old_expand_i}.double_conv.double_conv.0.weight'],  # oldModel_states[f'expand_blocks.{old_expand_i}.expand_block.double_conv.0.bias']],
                                    oldModel_states[f'expand_blocks.{old_expand_i}.double_conv.double_conv.3.weight'],]  # oldModel_states[f'expand_blocks.{old_expand_i}.expand_block.double_conv.3.bias']]]
                old_expand_i += 1
            self.expand_blocks.append(ExpandingBlock(options=options,
                                                     input_n=options['unet_conv_filters'][expand_n - 1],
                                                     output_n=options['unet_conv_filters'][expand_n - 2],
                                                     weights_list=weights_list_exp))

        if options['loss_sic'] == 'classification':
            self.sic_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0],
                                              output_n=options['n_classes']['SIC'],
                                              weights=[oldModel_states['sic_feature_map.feature_out.weight'],])
                                                    #    oldModel_states['sic_feature_map.feature_out.bias']])
        elif options['loss_sic'] == 'regression':
            # make the regression loss backwards compatible
            weights_regression = oldModel_states['sic_feature_map.feature_out.weight']
            print(f"Transfering a model with a SIC feature map of shape: {weights_regression.shape}")

            if weights_regression.shape[0] == options['n_classes']['SIC']:  # where the classes are outputted
                self.sic_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0],
                                                  output_n=1,)
                print(f"WARNING: Replaced the SIC feature map with a new one {self.sic_feature_map.state_dict} of shape: {list(self.sic_feature_map.state_dict().values())[0].shape}.")
            else:
                self.sic_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0],
                                                  output_n=1,
                                                  weights=[oldModel_states['sic_feature_map.feature_out.weight']])

        self.sod_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0],
                                          output_n=options['n_classes']['SOD'],
                                          weights=[oldModel_states['sod_feature_map.feature_out.weight'],])
                                                #    oldModel_states['sod_feature_map.feature_out.bias']])        
        self.floe_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0],
                                           output_n=options['n_classes']['FLOE'],
                                           weights=[oldModel_states['floe_feature_map.feature_out.weight']])

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))
        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1  

        return {'SIC': self.sic_feature_map(x_expand),
                'SOD': self.sod_feature_map(x_expand),
                'FLOE': self.floe_feature_map(x_expand)}


class FeatureMap(torch.nn.Module):
    """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

    def __init__(self, input_n, output_n, weights=None):
        super(FeatureMap, self).__init__()

        self.feature_out = torch.nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

        if weights is not None:
            with torch.no_grad():
                self.feature_out.weight = torch.nn.Parameter(weights[0].detach().clone())
                # self.feature_out.bias = torch.nn.Parameter(weights[1])

    def forward(self, x):
        """Pass x through final layer."""
        return self.feature_out(x)


class DoubleConv(torch.nn.Module):
    """Class to perform a double conv layer in the U-NET architecture. Used in unet_model.py."""

    def __init__(self, options, input_n, output_n, weights_list=None):
        super(DoubleConv, self).__init__()

        conv2d_layer1 = torch.nn.Conv2d(in_channels=input_n,
                                        out_channels=output_n,
                                        kernel_size=options['conv_kernel_size'],
                                        stride=options['conv_stride_rate'],
                                        padding=options['conv_padding'],
                                        padding_mode=options['conv_padding_style'],
                                        bias=False)
        conv2d_layer2 = torch.nn.Conv2d(in_channels=output_n,
                                        out_channels=output_n,
                                        kernel_size=options['conv_kernel_size'],
                                        stride=options['conv_stride_rate'],
                                        padding=options['conv_padding'],
                                        padding_mode=options['conv_padding_style'],
                                        bias=False)
            
        if weights_list is not None:
            with torch.no_grad():
                conv2d_layer1.weight = torch.nn.Parameter(weights_list[0].detach().clone())
                # conv2d_layer1.bias = torch.nn.Parameter(weights_list[0][1])
                conv2d_layer2.weight = torch.nn.Parameter(weights_list[1].detach().clone())
                # conv2d_layer2.bias = torch.nn.Parameter(weights_list[1][0])

        self.double_conv = torch.nn.Sequential(
            conv2d_layer1,
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU(),
            conv2d_layer2,
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU()
        )

    def forward(self, x):
        """Pass x through the double conv layer."""
        x = self.double_conv(x)

        return x


class ContractingBlock(torch.nn.Module):
    """Class to perform downward pass in the U-Net.
       For the weights:
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()"""

    def __init__(self, options, input_n, output_n, weights_list=None):
        super(ContractingBlock, self).__init__()

        self.contract_block = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.double_conv = DoubleConv(options, input_n, output_n, weights_list=weights_list)

    def forward(self, x):
        """Pass x through the downward layer."""
        x = self.contract_block(x)
        x = self.double_conv(x)
        return x


class ExpandingBlock(torch.nn.Module):
    """Class to perform upward layer in the U-Net."""

    def __init__(self, options, input_n, output_n, weights_list=None):
        super(ExpandingBlock, self).__init__()

        self.padding_style = options['conv_padding_style']
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.double_conv = DoubleConv(options, input_n=input_n + output_n, output_n=output_n, weights_list=weights_list)

    def forward(self, x, x_skip):
        """Pass x through the upward layer and concatenate with opposite layer."""
        x = self.upsample(x)

        # Insure that x and skip H and W dimensions match.
        x = expand_padding(x, x_skip, padding_style=self.padding_style)
        x = torch.cat([x, x_skip], dim=1)

        return self.double_conv(x)
    
    
def expand_padding(x, x_contract, padding_style: str = 'constant'):
    """
    Insure that x and x_skip H and W dimensions match.
    Parameters
    ----------
    x :
        Image tensor of shape (batch size, channels, height, width). Expanding path.
    x_contract :
        Image tensor of shape (batch size, channels, height, width) Contracting path.
        or torch.Size. Contracting path.
    padding_style : str
        Type of padding.

    Returns
    -------
    x : ndtensor
        Padded expanding path.
    """
    # Check whether x_contract is tensor or shape.
    if type(x_contract) == type(x):
        x_contract = x_contract.size()

    # Calculate necessary padding to retain patch size.
    pad_y = x_contract[2] - x.size()[2]
    pad_x = x_contract[3] - x.size()[3]

    if padding_style == 'zeros':
        padding_style = 'constant'

    x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

    return x

# class Attention_block(torch.nn.Module):
#     """Class for the attention block for u-nets.
#     Insprired by https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py"""

#     def __init__(self,F_g,F_l,F_int):
#         super(Attention_block,self).__init__()
#         self.W_g = torch.nn.Sequential(
#             torch.nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#             torch.nn.BatchNorm2d(F_int)
#             )
        
#         self.W_x = torch.nn.Sequential(
#             torch.nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#             torch.nn.BatchNorm2d(F_int)
#         )

#         self.psi = torch.nn.Sequential(
#             torch.nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
#             torch.nn.BatchNorm2d(1),
#             torch.nn.Sigmoid()
#         )
        
#         self.relu = torch.nn.ReLU(inplace=True)
        
#     def forward(self,g,x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1+x1)
#         psi = self.psi(psi)

#         return x*psi
