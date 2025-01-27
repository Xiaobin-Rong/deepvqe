# DeepVQE
A PyTorch implementation of DeepVQE described in [DeepVQE: Real Time Deep Voice Quality Enhancement for Joint Acoustic Echo Cancellation, Noise Suppression and Dereverberation](https://arxiv.org/pdf/2306.03177.pdf).

## About DeepVQE
![DeepVQE](https://github.com/Xiaobin-Rong/deepvqe/blob/main/pictures/DeepVQE.PNG)
DeepVQE is a speech enhancement (SE) model proposed by Microsoft for joint echo cancellation, noise suppression and dereverberation, which outperforms the top 1 models in both 2023 DNS Challenge and 2023 AEC Challenge.

DeepVQE utilizes the U-Net architecture as backbone, while makes some improvements:
* A new cross-attention mechanism for the microphone and far end soft alignment.
* Add residual block for each block in encoder and decoder.
* Use sub-pixel convolution instead of transposed convolution for up-sampling.
* A novel mask mechanism named complex convolving mask (CCM).

## Our purpose
We implement DeepVQE aiming to compare its SE performance with other two SOTA SE models, [DPCRN](https://arxiv.org/pdf/2107.05429.pdf) and [TF-GridNet](https://arxiv.org/pdf/2211.12433.pdf). To this end, We modify some experimental setup in the original paper, specifically:
* Datasets: we use DNS3 datasets in which all the utterances are sampled at 16 kHz.
* STFT: we use a squared root Hann window of length 32 ms, a hop length of 16 ms, and an FFT length of 512.
* Align Block: we drop the Align Block, because we do not focus on its AEC performance. Anyway, we still provide an implementation of the Align Block in our codes.

We are also interested in the inference speed presented in the paper, i.e, a relatively fast speed of 3.66 ms per frame in spite of its large complexity. So we also provide a stream version of DeepVQE, which is utilized to evaluate its inference speed.

## Requirements
einops <br/>
numpy<br/>
onnx<br/>
onnxruntime<br/>
onnxsim<br/>
ptflops<br/>
torch==1.11.0<br/>

## Results
### 1. SE performance
We are sorry to find that DeepVQE outperforms DPCRN only with a very limited margin, while requirng for much more computational resources (see below). Besides, DeepVQE lags behind TF-GridNet by a relatively large margin in terms of SE performance.
| Model    | Param. (M)| FLOPs (G) | OVRL / SIG/ BAK | DNSMOS-P.808 |
|:--------:|:---------:|:---------:|:---------------:|:------------:|
|DPCRN     |0.81       |3.73       |2.88 / 3.14 / 4/02|3.56         |
|TF-GridNet|1.60       |22.23      |2.98 / 3.25 / 4/05|3.65         |
|DeepVQE   |7.51       |8.04       |2.89 / 3.16 / 4.02|3.59         |


### 2. Inference speed 
We are surprised to find that although DeepVQE requires for large computational resources, it achieves a relatively good real-time factor of 0.2, which corresponds to the data presented in the paper. 
