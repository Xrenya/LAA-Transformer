# LAA-Transformer
Learning Attention from Attention: Efficient Self-Refinement Transformer for Face Super-Resolution, IJCAI 2023 (PyTorch Code)

**Unofficial model reproduction.**

[<img src="https://github.com/Guanxin-Li/LAA-Transformer/blob/main/architecture.png">](arch)

**TODO**:
- [ ] Build the model
  - Test replacing `DWT` layer with `F.conv2d(in_channels=in_channels, groups=in_channels)`
- [ ] Training Pipeline
- [ ] Train & test model


## Literate survey
1. [Learning Attention from Attention: Efficient Self-Refinement Transformer for Face Super-Resolution, IJCAI 2023](https://www.ijcai.org/proceedings/2023/0115.pdf)
2. [Wavelet Integrated CNNs for Noise-Robust Image Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.pdf)
3. [WaveCNet: Wavelet Integrated CNNs to Suppress Aliasing Effect for Noise-Robust Image Classification](https://arxiv.org/pdf/2107.13335.pdf)
4. [Memory-efficient Transformers via Top-k Attention](https://arxiv.org/pdf/2106.06899.pdf)

## Reference
1. [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)
2. [What is a haar wavelet?](https://www.collimator.ai/reference-guides/what-is-a-haar-wavelet)
3. [Top-k Attention](https://github.com/ag1988/top_k_attention/tree/main)
