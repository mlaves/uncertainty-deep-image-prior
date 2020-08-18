import torch
import torch.nn as nn
from .common import ProbabilityDropout2d, ProbabilityDropout, act, bn, conv, AddNoisyFMs


def decoder(
            num_input_channels=128, num_output_channels=3,
            num_channels=[16, 32, 64, 128, 128],
            num_channels_noise=[4, 4, 4, 4, 4],
            filter_size=3, need_sigmoid=True, need_bias=True,
            pad='zero', upsample_mode='bilinear', act_fun='LeakyReLU',
            need1x1=False, bayes=False, dm='2d', dp=0.5, dm_output='None', dp_output=0.5):
            """
            Assembles decoder only.

            Arguments:
                act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
                pad (string): zero|reflection (default: 'zero')
                upsample_mode (string): 'nearest|bilinear' (default: 'bilinear')
            """

            n_scales = len(num_channels)

            if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
                upsample_mode   = [upsample_mode] * n_scales

            if not (isinstance(filter_size, list) or isinstance(filter_size, tuple)) :
                filter_size   = [filter_size] * n_scales

            model = nn.Sequential()


            if num_channels[0] < num_channels[-1]:
                num_channels.reverse()
                num_channels_noise.reverse()

            for i in range(len(num_channels)):

                module_name_iter = i + 1

                if i != 0:
                    num_input_channels += num_channels_noise[i-1]
                # stride = 1 -- DIP: stride = 2
                model.add_module('conv_block_%d' % module_name_iter, conv(num_input_channels, num_channels[i], 3, bias=need_bias, pad=pad, dropout_mode=dm, dropout_p=dp, bayes=bayes, iterator=module_name_iter, string='dec'))
                model.add_module('bn_%d' % module_name_iter, bn(num_channels[i]))
                model.add_module('act_%d' % module_name_iter, act(act_fun))

                model.add_module('upsample_%d' % module_name_iter, nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

                if need1x1:
                    model.add_module('conv_block_1x1_%d' % module_name_iter, conv(num_channels[i], num_channels[i], 3, bias=need_bias, pad=pad, dropout_mode=dm, dropout_p=dp, bayes=bayes, iterator=module_name_iter, string='1x1'))
                    model.add_module('bn_1x1_%d' % module_name_iter, bn(num_channels[i]))
                    model.add_module('act_1x1_%d_1' % module_name_iter, act(act_fun))

                if num_channels_noise[i] != 0:
                    model.add_module('add_noisy_fms_%d' % module_name_iter, AddNoisyFMs(num_channels_noise[i]))

                num_input_channels = num_channels[i]

            final_iter = len(num_channels) + 1

            num_input_channels += num_channels_noise[i-1]
            
            model.add_module('conv_block_out', conv(num_input_channels, num_output_channels, 3, bias=need_bias, pad=pad, dropout_mode=dm_output, dropout_p=dp_output, bayes=bayes, iterator=final_iter, string='out'))
            if need_sigmoid:
                model.add_module('sigmoid', nn.Sigmoid())

            return model
