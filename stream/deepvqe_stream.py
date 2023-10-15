import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from modules.convolution import StreamConv2d
from modules.convert import convert_to_stream


class FE(nn.Module):
    """Feature extraction"""
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c
    def forward(self, x):
        """x: (B,F,1,2)"""
        x_mag = torch.sqrt(x[...,[0]]**2 + x[...,[1]]**2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1-self.c) + 1e-12)  # (B,F,T,2)
        return x_c.permute(0,3,2,1).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = StreamConv2d(channels, channels, kernel_size=(4,3), padding=(0,1))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()
    def forward(self, x, cache):
        """
        x: (B,C,1,F)
        cache: (B,C,3,F)
        """
        y, cache = self.conv(x, cache)
        y = self.elu(self.bn(y))
        return y + x, cache
    

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), stride=(1,2)):
        super().__init__()
        self.conv = StreamConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(0,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.resblock = ResidualBlock(out_channels)
    def forward(self, x, conv_cache, res_cache):
        """
        x: (B,C,1,F)
        conv_cache: (B,Ci,3,Fi)
        res_cache:  (B,Co,3,Fo)
        """
        x, conv_cache = self.conv(x, conv_cache)
        x = self.elu(self.bn(x))
        x, res_cache = self.resblock(x, res_cache)
        return x, conv_cache, res_cache


class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, cache):
        """x : (B,C,1,F)"""
        y = rearrange(x, 'b c t f -> b t (c f)')
        y, cache = self.gru(y, cache)
        y = self.fc(y)
        y = rearrange(y, 'b t (c f) -> b c t f', f=x.shape[-1])
        return y, cache
    

class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3)):
        super().__init__()
        self.conv = StreamConv2d(in_channels, out_channels*2, kernel_size, padding=(0,1))
        
    def forward(self, x, cache):
        """
        x: (B,C,1,F)
        cache: (B,C,3,F)
        """
        y, cache = self.conv(x, cache)
        y = rearrange(y, 'b (r c) t f -> b c t (r f)', r=2)
        return y, cache
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), is_last=False):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.resblock = ResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.is_last = is_last
    def forward(self, x, x_en, conv_cache, res_cache):
        """
        x: (B,C,1,F)
        x_en: (B,C,1,F)
        conv_cache: (B,Ci,2,Fi)
        res_cache: (B,Ci,2,Fi)
        """
        y = x + self.skip_conv(x_en)
        y, res_cache = self.resblock(y, res_cache)
        y, conv_cache = self.deconv(y, conv_cache)
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y, conv_cache, res_cache
    

class CCM(nn.Module):
    """Complex convolving mask block"""
    def __init__(self):
        super().__init__()
        # self.v = torch.tensor([1, -1/2 + 1j*np.sqrt(3)/2, -1/2 - 1j*np.sqrt(3)/2], dtype=torch.complex64)
        self.v = torch.tensor([[1,        -1/2,           -1/2],
                               [0, np.sqrt(3)/2, -np.sqrt(3)/2]], dtype=torch.float32)  # (2,3)
        self.unfold = nn.Unfold(kernel_size=(3,3), padding=(0,1))
    
    def forward(self, m, x, cache):
        """
        m: (B,27,1,F)
        x: (B,F,1,2)
        cache: (B,F,2,2)
        """
        m = m.view(1, 3, 9, 1, 257)
        H_real = torch.sum(self.v[0].to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)
        H_imag = torch.sum(self.v[1].to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)
        
        M_real = H_real.view(1, 3, 3, 1, 257)
        M_imag = H_imag.view(1, 3, 3, 1, 257)
        
        x = torch.cat([cache, x], dim=2)     # (B,F,T,2)
        cache = x[:,:,1:]                    # (B,F,2,2)
        x = x.permute(0,3,2,1).contiguous()  # (B,2,T,F)

        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(1, 2, 3, 3, 1, 257)
        
        x_enh_real = torch.sum(M_real * x_unfold[:,0] - M_imag * x_unfold[:,1], dim=(1,2))  # (B,T,F)
        x_enh_imag = torch.sum(M_real * x_unfold[:,1] + M_imag * x_unfold[:,0], dim=(1,2))  # (B,T,F)
        x_enh = torch.stack([x_enh_real, x_enh_imag], dim=3).transpose(1,2).contiguous()

        return x_enh, cache


class StreamDeepVQE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = FE()
        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)
        
        self.bottle = Bottleneck(128*9, 64*9)
        
        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 27)
        self.ccm = CCM()
        
    def forward(self, x, en_conv_cache1, en_res_cache1, en_conv_cache2, en_res_cache2, en_conv_cache3, en_res_cache3,
                en_conv_cache4, en_res_cache4, en_conv_cache5, en_res_cache5,
                h_cache, de_conv_cache5, de_res_cache5, de_conv_cache4, de_res_cache4, de_conv_cache3, de_res_cache3,
                de_conv_cache2, de_res_cache2, de_conv_cache1, de_res_cache1,
                m_cache):
        """
        x             : (B,257,1,2)
        en_conv_cache1: (1, 2,3,257)
        en_res_cache1 : (1,64,3,129)
        en_conv_cache2: (1,64,3,129)
        en_res_cache2 : (1,128,3,65)
        en_conv_cache3: (1,128,3,65)
        en_res_cache3 : (1,128,3,33)
        en_conv_cache4: (1,128,3,33)
        en_res_cache4 : (1,128,3,17)
        en_conv_cache5: (1,128,3,17)
        en_res_cache5 : (1,128,3,9)
        h_cache       : (1,1,64*9)
        de_res_cache5 : (1,128,3,9)
        de_conv_cache5: (1,128,3,9)
        de_res_cache4 : (1,128,3,17)
        de_conv_cache4: (1,128,3,17)
        de_res_cache3 : (1,128,3,33)
        de_conv_cache3: (1,128,3,33)
        de_res_cache2 : (1,128,3,65)
        de_conv_cache2: (1,128,3,65)
        de_res_cache1 : (1,64,3,129)
        de_conv_cache1: (1,64,3,129)
        m_cache       : (1,257,2, 2)
        """
        en_x0 = self.fe(x)            # ; print(en_x0.shape)
        en_x1, en_conv_cache1, en_res_cache1 = self.enblock1(en_x0, en_conv_cache1, en_res_cache1)
        en_x2, en_conv_cache2, en_res_cache2 = self.enblock2(en_x1, en_conv_cache2, en_res_cache2)
        en_x3, en_conv_cache3, en_res_cache3 = self.enblock3(en_x2, en_conv_cache3, en_res_cache3)
        en_x4, en_conv_cache4, en_res_cache4 = self.enblock4(en_x3, en_conv_cache4, en_res_cache4)
        en_x5, en_conv_cache5, en_res_cache5 = self.enblock5(en_x4, en_conv_cache5, en_res_cache5)

        en_xr, h_cache = self.bottle(en_x5, h_cache)

        de_x5, de_conv_cache5, de_res_cache5 = self.deblock5(en_xr, en_x5, de_conv_cache5, de_res_cache5)
        de_x5 = de_x5[..., :en_x4.shape[-1]]
        de_x4, de_conv_cache4, de_res_cache4 = self.deblock4(de_x5, en_x4, de_conv_cache4, de_res_cache4)
        de_x4 = de_x4[..., :en_x3.shape[-1]]
        de_x3, de_conv_cache3, de_res_cache3 = self.deblock3(de_x4, en_x3, de_conv_cache3, de_res_cache3)
        de_x3 = de_x3[..., :en_x2.shape[-1]]
        de_x2, de_conv_cache2, de_res_cache2 = self.deblock2(de_x3, en_x2, de_conv_cache2, de_res_cache2)
        de_x2 = de_x2[..., :en_x1.shape[-1]]
        de_x1, de_conv_cache1, de_res_cache1 = self.deblock1(de_x2, en_x1, de_conv_cache1, de_res_cache1)
        de_x1 = de_x1[..., :en_x0.shape[-1]]
        
        
        x_enh, m_cache = self.ccm(de_x1, x, m_cache)  # (B,F,T,2)
        
        return x_enh, en_conv_cache1, en_res_cache1, en_conv_cache2, en_res_cache2, en_conv_cache3, en_res_cache3,\
                en_conv_cache4, en_res_cache4, en_conv_cache5, en_res_cache5, \
                h_cache, de_conv_cache5, de_res_cache5, de_conv_cache4, de_res_cache4, de_conv_cache3, de_res_cache3,\
                de_conv_cache2, de_res_cache2, de_conv_cache1, de_res_cache1, m_cache


if __name__ == "__main__":
    import time
    from deepvqe import DeepVQE
    
    
    device = "cpu"
    model = DeepVQE().eval().to(device)
    stream_model = StreamDeepVQE().eval().to(device)

    convert_to_stream(stream_model, model)

    batch = torch.randn(1, 257, 100, 2, device=device)

    """非流式推理"""
    output = model(batch)

    """流式推理"""
    en_conv_cache1 = torch.zeros(1,2,3,257, device=device)
    en_res_cache1  = torch.zeros(1,64,3,129, device=device)
    en_conv_cache2 = torch.zeros(1,64,3,129, device=device)
    en_res_cache2  = torch.zeros(1,128,3,65, device=device)
    en_conv_cache3 = torch.zeros(1,128,3,65, device=device)
    en_res_cache3  = torch.zeros(1,128,3,33, device=device)
    en_conv_cache4 = torch.zeros(1,128,3,33, device=device)
    en_res_cache4  = torch.zeros(1,128,3,17, device=device)
    en_conv_cache5 = torch.zeros(1,128,3,17, device=device)
    en_res_cache5  = torch.zeros(1,128,3,9, device=device)
    h_cache        = torch.zeros(1,1,64*9, device=device)
    de_res_cache5  = torch.zeros(1,128,3,9, device=device)
    de_conv_cache5 = torch.zeros(1,128,3,9, device=device)
    de_res_cache4  = torch.zeros(1,128,3,17, device=device)
    de_conv_cache4 = torch.zeros(1,128,3,17, device=device)
    de_res_cache3  = torch.zeros(1,128,3,33, device=device)
    de_conv_cache3 = torch.zeros(1,128,3,33, device=device)
    de_res_cache2  = torch.zeros(1,128,3,65, device=device)
    de_conv_cache2 = torch.zeros(1,128,3,65, device=device)
    de_res_cache1  = torch.zeros(1,64,3,129, device=device)
    de_conv_cache1 = torch.zeros(1,64,3,129, device=device)
    m_cache        = torch.zeros(1,257,2,2, device=device)
    

    times = []
    outputs = []
    for i in range(batch.shape[-2]):
        x = batch[:,:,i:i+1,:]
        
        tic = time.perf_counter()
        y, en_conv_cache1, en_res_cache1, en_conv_cache2, en_res_cache2, en_conv_cache3, en_res_cache3,\
                    en_conv_cache4, en_res_cache4, en_conv_cache5, en_res_cache5, \
                    h_cache, de_conv_cache5, de_res_cache5, de_conv_cache4, de_res_cache4, de_conv_cache3, de_res_cache3,\
                    de_conv_cache2, de_res_cache2, de_conv_cache1, de_res_cache1, m_cache \
        = stream_model(x, en_conv_cache1, en_res_cache1, en_conv_cache2, en_res_cache2, en_conv_cache3, en_res_cache3,
                        en_conv_cache4, en_res_cache4, en_conv_cache5, en_res_cache5,
                        h_cache, de_conv_cache5, de_res_cache5, de_conv_cache4, de_res_cache4, de_conv_cache3, de_res_cache3,
                        de_conv_cache2, de_res_cache2, de_conv_cache1, de_res_cache1,
                        m_cache)
        toc = time.perf_counter()

        times.append((toc-tic)*1000)
        outputs.append(y)

    outputs = torch.cat(outputs, dim=-2)
    times = times[1:]
    print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(sum(times)/len(times), max(times), min(times)))
    print(">>> Streaming error:", (output - outputs).abs().max().item())


    """ONNX模型"""
    import time
    import onnx
    import onnxruntime
    from onnxsim import simplify
    ## convert to onnx
    file = 'onnx_models/deepvqe.onnx'

    input = torch.randn(1, 257, 1, 2, device=device)
    torch.onnx.export(stream_model,
                    (input, en_conv_cache1, en_res_cache1, en_conv_cache2, en_res_cache2, en_conv_cache3, en_res_cache3,
                    en_conv_cache4, en_res_cache4, en_conv_cache5, en_res_cache5,
                    h_cache, de_conv_cache5, de_res_cache5, de_conv_cache4, de_res_cache4, de_conv_cache3, de_res_cache3,
                    de_conv_cache2, de_res_cache2, de_conv_cache1, de_res_cache1, m_cache),
                    file,
                    input_names = ['mix', 'en_conv_cache1', 'en_res_cache1', 'en_conv_cache2', 'en_res_cache2', 'en_conv_cache3', 'en_res_cache3',
                    'en_conv_cache4', 'en_res_cache4', 'en_conv_cache5', 'en_res_cache5',
                    'h_cache', 'de_conv_cache5', 'de_res_cache5', 'de_conv_cache4', 'de_res_cache4', 'de_conv_cache3', 'de_res_cache3',
                    'de_conv_cache2', 'de_res_cache2', 'de_conv_cache1', 'de_res_cache1', 'm_cache'],
                    output_names = ['enh', 'en_conv_cache1_out', 'en_res_cache1_out', 'en_conv_cache2_out', 'en_res_cache2_out', 'en_conv_cache3_out', 'en_res_cache3_out',
                    'en_conv_cache4_out', 'en_res_cache4_out', 'en_conv_cache5_out', 'en_res_cache5_out',
                    'h_cache_out', 'de_conv_cache5_out', 'de_res_cache5_out', 'de_conv_cache4_out', 'de_res_cache4_out', 'de_conv_cache3_out', 'de_res_cache3_out',
                    'de_conv_cache2_out', 'de_res_cache2_out', 'de_conv_cache1_out', 'de_res_cache1_out', 'm_cache_out'],
                    opset_version=11,
                    verbose = False)

    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model)

    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, file.split('.onnx')[0] + '_simple.onnx')

    ## run onnx model
    # session = onnxruntime.InferenceSession(file, None, providers=['CPUExecutionProvider'])
    session = onnxruntime.InferenceSession(file.split('.onnx')[0]+'_simple.onnx', None, providers=['CPUExecutionProvider'])
    en_conv_cache1 = np.zeros([1,2,3,257],  dtype="float32")
    en_res_cache1  = np.zeros([1,64,3,129], dtype="float32")
    en_conv_cache2 = np.zeros([1,64,3,129], dtype="float32")
    en_res_cache2  = np.zeros([1,128,3,65], dtype="float32")
    en_conv_cache3 = np.zeros([1,128,3,65], dtype="float32")
    en_res_cache3  = np.zeros([1,128,3,33], dtype="float32")
    en_conv_cache4 = np.zeros([1,128,3,33], dtype="float32")
    en_res_cache4  = np.zeros([1,128,3,17], dtype="float32")
    en_conv_cache5 = np.zeros([1,128,3,17], dtype="float32")
    en_res_cache5  = np.zeros([1,128,3,9 ], dtype="float32")
    h_cache        = np.zeros([1,1,64*9  ], dtype="float32")
    de_res_cache5  = np.zeros([1,128,3,9 ], dtype="float32")
    de_conv_cache5 = np.zeros([1,128,3,9 ], dtype="float32")
    de_res_cache4  = np.zeros([1,128,3,17], dtype="float32")
    de_conv_cache4 = np.zeros([1,128,3,17], dtype="float32")
    de_res_cache3  = np.zeros([1,128,3,33], dtype="float32")
    de_conv_cache3 = np.zeros([1,128,3,33], dtype="float32")
    de_res_cache2  = np.zeros([1,128,3,65], dtype="float32")
    de_conv_cache2 = np.zeros([1,128,3,65], dtype="float32")
    de_res_cache1  = np.zeros([1,64,3,129], dtype="float32")
    de_conv_cache1 = np.zeros([1,64,3,129], dtype="float32")
    m_cache        = np.zeros([1,257,2,2],  dtype="float32")

    T_list = []
    outputs = []

    inputs = batch.numpy()
    for i in range(inputs.shape[-2]):
        tic = time.perf_counter()
        
        out_i,  en_conv_cache1, en_res_cache1, en_conv_cache2, en_res_cache2, en_conv_cache3, en_res_cache3,\
                en_conv_cache4, en_res_cache4, en_conv_cache5, en_res_cache5,\
                h_cache, de_conv_cache5, de_res_cache5, de_conv_cache4, de_res_cache4, de_conv_cache3, de_res_cache3,\
                de_conv_cache2, de_res_cache2, de_conv_cache1, de_res_cache1, m_cache\
                = session.run([], {'mix': inputs[..., i:i+1, :],
                    'en_conv_cache1': en_conv_cache1, 'en_res_cache1': en_res_cache1, 
                    'en_conv_cache2': en_conv_cache2, 'en_res_cache2': en_res_cache2, 
                    'en_conv_cache3': en_conv_cache3, 'en_res_cache3': en_res_cache3,
                    'en_conv_cache4': en_conv_cache4, 'en_res_cache4': en_res_cache4, 
                    'en_conv_cache5': en_conv_cache5, 'en_res_cache5': en_res_cache5,
                    'h_cache': h_cache, 
                    'de_conv_cache5': de_conv_cache5, 'de_res_cache5': de_res_cache5, 
                    'de_conv_cache4': de_conv_cache4, 'de_res_cache4': de_res_cache4, 
                    'de_conv_cache3': de_conv_cache3, 'de_res_cache3': de_res_cache3,
                    'de_conv_cache2': de_conv_cache2, 'de_res_cache2': de_res_cache2, 
                    'de_conv_cache1': de_conv_cache1, 'de_res_cache1': de_res_cache1,
                    'm_cache': m_cache})

        toc = time.perf_counter()
        T_list.append(toc-tic)
        outputs.append(out_i)

    print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(1e3*np.mean(T_list), 1e3*np.max(T_list), 1e3*np.min(T_list)))

    outputs = np.concatenate(outputs, axis=-2)
    print(">>> Onnx error:", np.abs(output.detach().cpu().numpy() - outputs).max())




