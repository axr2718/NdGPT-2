# NdGPT-2: Replacing nn.Linear with NdLinear in GPT-2.
## Introduction
This repo contains a PyTorch implementation of [GPT-2](https://github.com/openai/gpt-2) from scratch using the [NdLinear](https://arxiv.org/pdf/2503.17353) layer, which is a more efficient and less compute-heavy replacement for nn.Linear. Using NdLinear, we can show that the model, NdGPT-2, surpasses or achieves the same performance as GPT-2 124M despite having less parameters. For every block in the transformer layers, I replace nn.Linear with NdLinear in all the MLP blocks. 


NdGPT-2 is trained on [FineWeb-Edu](https://arxiv.org/pdf/2503.17353).
