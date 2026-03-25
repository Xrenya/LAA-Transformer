# LAA-Transformer
Learning Attention from Attention: Efficient Self-Refinement Transformer for Face Super-Resolution, IJCAI 2023 (PyTorch Code)

**Unofficial model reproduction.**

[<img src="https://github.com/Guanxin-Li/LAA-Transformer/blob/main/architecture.png">](arch)

**TODO**:
- [x] Build the model (the forward pass works, but still not sure that the implentation is exact due to out of memory with image size 512x512
  - Haar based model
  - Conv haar based model (tried to fix out of memory issue)
  - **Notes**: dwt.py, idwt.py, model.py have some issues in the implementation but I will leave it as it is. 
- [ ] Training Pipeline
- [ ] Train & test model


## Literature survey
1. [Learning Attention from Attention: Efficient Self-Refinement Transformer for Face Super-Resolution, IJCAI 2023](https://www.ijcai.org/proceedings/2023/0115.pdf)
2. [Wavelet Integrated CNNs for Noise-Robust Image Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.pdf)
3. [WaveCNet: Wavelet Integrated CNNs to Suppress Aliasing Effect for Noise-Robust Image Classification](https://arxiv.org/pdf/2107.13335.pdf)
4. [Memory-efficient Transformers via Top-k Attention](https://arxiv.org/pdf/2106.06899.pdf)

## References
1. [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)
2. [What is a haar wavelet?](https://www.collimator.ai/reference-guides/what-is-a-haar-wavelet)
3. [Top-k Attention](https://github.com/ag1988/top_k_attention/tree/main)
