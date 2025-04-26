# 1. Introduction
Comparing GPT-2 with and without [pruning](https://arxiv.org/abs/2306.11695) and with and without [NdLinear](https://arxiv.org/abs/2503.17353) for EnsembleAI.

# 2. Model Summary
This repo contains GPT-2 (124M) from scratch, modified and edited from this [repo](https://github.com/karpathy/build-nanogpt/tree/master). The model is trained on [FineWeb-Edu](https://github.com/karpathy/build-nanogpt/tree/master) `sample-10BT` and evaluated on [HellaSwag](https://paperswithcode.com/dataset/hellaswag).

# 3. Repo Layout
`experiment` contains the training, testing, and text generation files for GPT-2.


`models` contains the GPT-2 model, along with its DataLoader.


`utils` contains helper functions for HellaSwag, setting seed for reproducibility, and parameter counts.


`logs` contains the train, validation, and HellaSwag evaluation for different models.


`config.py` contains the GPT-2 and training hyper/parameters.


`main.py` contains a simple test run.


`run.ipynb` will contain all the results.

