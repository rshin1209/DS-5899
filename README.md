# Flash Attention
This document was prepared **only** for DS-5899 Paper Presentation at Vanderbilt University.
The publication (FLASHATTENTION: Fast and Memory-Efficient Exact Attention with IO-Awareness) by Dao et al. can be found [here](https://arxiv.org/abs/2205.14135)
## Motivation of Flash Attention: Modeling Longer Sequences
* **NLP**: Large context required to understand books, plays, and instruction manuals
* **Computer Vision**: Higher resolution can lead to a better and more robust insight
* **Time series, audio, video, medical imaging**: Data are intrinsicly modeled as sequences of multiple steps

## Attention is the HEART of Transformers
<img src="https://user-images.githubusercontent.com/25111091/197630239-df4a88d6-7bd6-4d81-88cd-f3beae23fb9e.png" width="500">

### Question 1: What are the existing methods to accomodate longer sequences?
## Existing studies for the 
**Existing studies are APPROXIMATE attention and do not show wall-clock speedup against the standard attention method.**
### Sparse-approximations
* Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In The International Conference on Machine Learning (ICML), 2020.
* Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9: 53–68, 2021.
### Low-rank approximations
* Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In International Conference on Learning Representations (ICLR), 2020.
* Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, pages 5156–5165. PMLR, 2020.
* Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.
### Sparse and Low-rank Approximations
* Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.
* Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Ré. Scatterbrain: Unifying sparse and low-rank attention. In Advances in Neural Information Processing Systems (NeurIPS), 2021.
* Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33, 2020.

## Background: Attention Layer makes Transformers slow and memory-hungry on long sequences.
![Screen Shot 2022-10-24 at 4 12 20 PM](https://user-images.githubusercontent.com/25111091/197630379-74042ca2-a8f1-4c29-b029-c4e7019a79f7.png)
### **O** = Dropout(Softmax(Mask(**QK** <sup>**T**</sup>)))**V**
### Question 2: Is Attention **Compute-Bound** or **Memory-Bound**?

## Flash Attention
![Screen Shot 2022-10-24 at 4 34 05 PM](https://user-images.githubusercontent.com/25111091/197633996-2a1553f9-3126-4158-a964-b90911b5c660.png)
* **GPU Memory Hierarchy (Left)**: For example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s.
* **FlashAttention (Middle)**: FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM.
* **Attention Speed Comparison (Right)**: Speedup comparison against the PyTorch implementation of attention on GPT-2.
### Question 3: FlashAttention involves "tiling" and "recomputation" techniques. Why does it need recomputation?
## Standard Attention Implementation
![Screen Shot 2022-10-24 at 4 16 04 PM](https://user-images.githubusercontent.com/25111091/197633381-886b30ad-027d-4fde-8862-260fc79d477d.png)
* O(Nd+N<sup>2</sup>)
## Flash Attention Implementation
![Screen Shot 2022-10-24 at 4 13 45 PM](https://user-images.githubusercontent.com/25111091/197630869-d6a48fba-d4f1-4027-ae48-7fc3a4a820ad.png)
* O(N<sup>2</sup>d<sup>2</sup>M<sup>-1</sup>)
## Flash Attention Benchmark
![Screen Shot 2022-10-24 at 4 36 00 PM](https://user-images.githubusercontent.com/25111091/197634326-b64e78b8-1879-4fbb-ae9b-895d01b4cb4c.png)
