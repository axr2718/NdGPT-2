# Introducing NdGPT-2

This model is GPT-2 from scratch by OpenAI with [NdLinear](https://arxiv.org/abs/2503.17353) from [Ensemble AI](https://ensemblecore.ai/). I used [GPT-3 hyperparameters](https://arxiv.org/abs/2005.14165) for this project. The model is quantized using bfloat16 and tf32.


GPT-2 from OpenAI has 124M parameters, but using NdLinear, I reduced this count to 96M, which is nearly a 25% reduction. I went pretty aggressive on the parameter reduction due to time limit constraints, so I don't expect the model to surpass GPT-2 or GPT-3 in my short run.


I am using [FineWebEDU-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb) as the dataset for training, which contains 10 billion tokens of high quality educational data. There was no time to fine tune it on answer/question datasets for an agent.


Every 250 steps, the model is evaluated on the validation dataset and generates 4 samples of text, and that specific model is saved. We also evaluate on the [HellaSwag](https://rowanzellers.com/hellaswag/) benchmark to see how it compares to other models.


After training is done, we print out some statistics to compare it to GPT-2 and GPT-3 models.


Lastly, we prune each model with [Wanda](https://arxiv.org/abs/2306.11695) and compare the number of parameters and one-shot performance on HellaSwag.

## Repo Layout
`experiment` contains the text generation, GPT training, and GPT testing (validation, HellaSwag evaluation), and train/test for a ViT that is not used.


`models` contains GPT-2 from OpenAI, NdGPT-2, ViT, and NdViT. Note that both vision transformers were not used, but the code is there if you are interested in looking.

`pruning` contains the Wanda pruning algorithm code.


`utils` contains utilities for setting a seed, evaluating on HellaSwag, and counting parameters.


## Please Read
Unfortunately, I found the internship application pretty late (about a week ago), so I didn't have time to go full in on this project before the deadline due to courses and research projects, but I would have loved to do more if I had the time. Due to this, I was only able to train NdGPT-2 on half an epoch of the dataset because it would take my computer around ~9 days for a full epoch. However, despite this, the model loss still goes down and the evaluations are on-par with normal GPT-2 for half an epoch. Originally I was going to train both GPT-2 and NdGPT-2 for 1 epoch, but that would take over 2 weeks.