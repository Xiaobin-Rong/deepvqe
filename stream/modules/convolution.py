# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:32:08 2022

@author: lexiaohuai
"""

import torch.nn as nn
import torch
from typing import Optional, List, Tuple, Union
import numpy as np
"""
注意cache导出onnx的时候不能保存成list，要以tensor输出
"""

class StreamConv1d(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int=1,
                padding: int=0,
                dilation: int=1,
                groups: int=1,
                bias: bool=True,
                *args, **kargs):
        super(StreamConv1d, self).__init__(*args, *kargs)
        self.Conv1d = nn.Conv1d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation = dilation,
                                groups = groups,
                                bias = bias)
    
    def forward(self, x, cache):
        """
        x:     [bs, C, T_size]
        cache: [bs, C, T_size-1]
        """
        inp = torch.cat([cache, x], dim=-1)
        oup = self.Conv1d(inp)
        out_cache = inp[..., 1:]
        return oup, out_cache


class StreamConv2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 *args, **kargs):
        super().__init__(*args, **kargs)
        """
        流式卷积实现。
        默认 kernel_size = [T_size, F_size]
        """
        self.Conv2d = nn.Conv2d(in_channels = in_channels, 
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation = dilation,
                                groups = groups,
                                bias = bias)
            
    def forward(self, x, cache):
        """
        x: [bs, C, 1, F]
        cache: [bs, C, T_size-1, F]
        """
        inp = torch.cat([cache, x], dim=2)
        outp = self.Conv2d(inp)
        out_cache = inp[:,:, 1:]
        return outp, out_cache


def stream_conv1d_forward(self, x, cache):
    inp = torch.cat([cache, x], dim=-1)
    oup = nn.functional.conv1d(inp, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
    out_cache = inp[..., 1:]
    return oup, out_cache


class StreamConvTranspose2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 *args, **kargs):
        super().__init__(*args, **kargs)
        """
        流式转置卷积实现。
        默认 kernel_size = [T_size, F_size]
        默认 stride = [T_stride, F_stride] 且 T_stride == 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(kernel_size) is int:
            self.T_size = kernel_size
            self.F_size = kernel_size
        elif type(kernel_size) in [list, tuple]:
            self.T_size, self.F_size = kernel_size
        else:
            raise ValueError('Invalid kernel size.')
            
        if type(stride) is int:
            self.T_stride = stride
            self.F_stride = stride
        elif type(stride) in [list, tuple]:
            self.T_stride, self.F_stride = stride
        else:
            raise ValueError('Invalid stride size.')
        
        assert self.T_stride == 1

        if type(padding) is int:
            self.T_pad = padding
            self.F_pad = padding
        elif type(padding) in [list, tuple]:
            self.T_pad, self.F_pad = padding
        else:
            raise ValueError('Invalid padding size.')
        assert(self.T_pad == 0) 

        if type(dilation) is int:
            self.T_dilation = dilation
            self.F_dilation = dilation
        elif type(dilation) in [list, tuple]:
            self.T_dilation, self.F_dilation = dilation
        else:
            raise ValueError('Invalid dilation size.')

        assert groups == 1
        
        # 我们使用权重时间反向的Conv2d实现转置卷积    
        self.ConvTranspose2d = nn.Conv2d(in_channels = in_channels, 
                                        out_channels = out_channels,
                                        kernel_size = kernel_size,
                                        stride = (self.T_stride, 1), # 若F维度stride不为1，将在forward中使用额外的上采样算子
                                        padding = (self.T_pad, 0),   # 若F维度padding不为0，将在forward中使用额外的填充
                                        dilation = dilation,
                                        groups = groups,
                                        bias = bias)
        
    def forward(self, x, cache):
        """
        x: [bs,C,1,F]
        cache: [bs,C,T-1,F]
        """
        # [bs,C,T,F]
        inp = torch.cat([cache, x], dim = 2)
        out_cache = inp[:, :, 1:]
        bs, C, T, F = inp.shape
        #添加上采样算子
        if self.F_stride > 1: 
            # [bs,C,T,F] -> [bs,C,T,F,1] -> [bs,C,T,F,F_stride] -> [bs,C,T,F_out]
            inp = torch.cat([inp[:,:,:,:,None], torch.zeros([bs,C,T,F,self.F_stride-1],device=x.device)], dim = -1).reshape([bs,C,T,-1])
            left_pad = self.F_stride - 1
            if self.F_size > 1:
                if left_pad <= self.F_size - 1:
                    inp = torch.nn.functional.pad(inp, pad = [(self.F_size - 1)*self.F_dilation-self.F_pad, (self.F_size - 1)*self.F_dilation-self.F_pad - left_pad, 0, 0])
                else:
                    # inp = torch.nn.functional.pad(inp, pad = [self.F_size - 1, 0, 0, 0])[:,:,:,: - (left_pad - self.F_stride + 1)]
                    raise(NotImplementedError)
            else:
                # inp = inp[:,:,:,:-left_pad]
                raise(NotImplementedError)

        else: # F_stride = 1
            inp = torch.nn.functional.pad(inp, pad=[(self.F_size-1)*self.F_dilation-self.F_pad, (self.F_size-1)*self.F_dilation-self.F_pad])
                
        outp = self.ConvTranspose2d(inp)
    
        return outp, out_cache



if __name__ == '__main__':
    import types
    from convert import convert_to_stream

    torch.random.seed()

    # test Conv1d Stream
    # SC = StreamConv1d(1, 1, 3)
    # Conv = nn.Conv1d(1, 1, 3)
    # convert_to_stream(SC, Conv)

    # test_input = torch.randn([1, 1, 10])
    # with torch.no_grad():
    #     # Non-Streaming
    #     test_out1 = Conv(torch.nn.functional.pad(test_input, [2,0]))
        
    #     # Streaming
    #     cache = torch.zeros([1, 1, 2])
    #     test_out2 = []
    #     for i in range(10):
    #         out, cache = SC(test_input[..., i:i+1], cache)
    #         test_out2.append(out)
    #     test_out2 = torch.cat(test_out2, dim=-1)
    #     print((test_out1 - test_out2).abs().max())

    #     # Streaming Method 2
    #     Conv.forward = types.MethodType(stream_conv1d_forward, Conv)
    #     cache = torch.zeros([1, 1, 2])
    #     test_out3 = []
    #     for i in range(10):
    #         out, cache = Conv(test_input[..., i:i+1], cache)
    #         test_out3.append(out)
    #     test_out3 = torch.cat(test_out3, dim=2)
    #     print((test_out1 - test_out3).abs().max())

    # # test Conv2d Stream
    # SC = StreamConv2d(1, 1, [3,3])
    # Conv = nn.Conv2d(1, 1, (3,3))
    # convert_to_stream(SC, Conv)

    # test_input = torch.randn([1,1,10,6])

    # with torch.no_grad():
    #     # Non-Streaming
    #     test_out1 = Conv(torch.nn.functional.pad(test_input,[0,0,2,0]))
        
    #     # Streaming
    #     cache = torch.zeros([1,1,2,6])
    #     test_out2 = []
    #     for i in range(10):
    #         out, cache = SC(test_input[:,:, i:i+1], cache)
    #         test_out2.append(out)
    #     test_out2 = torch.cat(test_out2, dim=2)
    #     print((test_out1 - test_out2).abs().max())


    # test ConvTranspose2d Stream
    DeConv = torch.nn.ConvTranspose2d(4, 8, (3,3), (1,2), padding=(0,1), dilation=(1,4), groups=1)
    SDC = StreamConvTranspose2d(4, 8, (3,3), (1,2), padding=(0,1), dilation=(1,4), groups=1)
    convert_to_stream(SDC, DeConv)

    test_input = torch.randn([1,4,10,6])
    with torch.no_grad():
        # Non-Streaming
        test_out1 = DeConv(test_input)
        test_out1 = test_out1[:,:, :10]
        # Streaming
        test_out2 = []
        cache = torch.zeros([1,4,2,6])
        for i in range(10):
            out, cache = SDC(test_input[:,:, i:i+1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=2)

        print(test_out1.shape)
        print(test_out2.shape)
        print((test_out1 - test_out2).abs().max())
    

