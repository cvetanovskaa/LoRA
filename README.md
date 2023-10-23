# LoRA: Low-Rank Adaptation of Large Language Models
This repository contains overview, explanation, and examples of LoRA as outlined in the official paper: https://arxiv.org/abs/2106.09685

Wang, S., Shen, Y., Liu, J., Chen, P.-Y., Chen, M., Bernstein, M., & Li, J. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685

# Introduction
Large language models (LLMs) like GPT-4 have demonstrated remarkable capabilities across a wide range of natural language processing tasks. However, to achieve optimal performance on specific tasks or domains, these models often require fine-tuning, which involves adjusting the model's parameters based on task-specific data.
Despite its importance, fine-tuning LLMs can be highly resource-intensive. Traditional fine-tuning approaches involve updating all the parameters of the model, which can be prohibitively expensive in terms of computational resources and time.
<br /><br /><br />
![image](https://github.com/cvetanovskaa/LoRA/assets/15224551/ddc5fc69-824d-4e98-a950-70bb728927d8)

With the ever-growing size of these models, one critical question has come to the forefront:
**"How can we efficiently train LLMs to maintain high-quality performance for specific tasks, while minimizing computational costs and avoiding additional inference latency?"**

Enter LoRA - a novel approach designed to address this very challenge by streamlining the fine-tuning process while preserving the model's efficacy and introducing zero inference latency. 

# Paper Overview
## Key Idea
LoRA adapts the weight matrices of attention layers in LLMs by introducing low-rank decomposed trainable matrices, which significantly reduces the number of parameters to be trained during the fine-tuning process. This approach targets the attention mechanism of transformers, which is a critical component for capturing long-range dependencies in text.

## Rank
Imagine you have a bookshelf with some rows filled with books. The rank of this bookshelf is how many rows are not empty. If some rows have the same books as others, we don't count them. So, the rank tells us how many rows have different books. Similarly, in a matrix, the rank tells us how many rows (or columns) are linearly independent, i.e have unique information, not repeated elsewhere.

If a matrix has low rank compared to its size, that means there is a lot of repetition and redundancy in the data it holds. Many of its rows (or columns) are linear combinations of others and can be derived from them.

**Hypothesis: The weight change metrices, used during model adaptation, have low "intrinsic rank," i.e. they can be described almost as accurately with fewer dimensions than the models originally have.**

## Low-Rank Decomposition
What is low-rank decomposition? A technique used to approximate a matrix with a product of two or more matrices that have lower dimensions compared to the original matrix. The key advantage of this decomposition is that it significantly reduces the number of trainable parameters.

### Practical Example:
- Transformer weights (according to "Attention is all you need" paper), have dimensions of d x k = 512 x 64 = 32,768 trainable parameters.
- In LoRA with rank 8, we have two matrices:
  - 512 x 8 = 4096 parameters
  - 8 x 64 = 512 parameters
  - Therefore, LoRA gives us 4608 trainable parameteres => 86% reduction

<p float="left">
  <img src="https://github.com/cvetanovskaa/LoRA/assets/15224551/7846c998-c92e-4f47-9bb2-ded02045924f" width="500" />
  <img src="https://github.com/cvetanovskaa/LoRA/assets/15224551/ea70ad34-87b9-426b-b494-0f3266c40b70" width="500" />
</p>

## LoRA Finetuning Process

1. Initialization of low-rank matrices for each attention layer. The dimensions are A ∈ R^{r x d}, B ∈ R^{d x r} where r is the rank and d is the input dimension.
2. Freezing the original weight matrices in attention layers so they do not get updated during fine-tuning.
3. Iteratively updating only the low-rank matrices (A and B) based on gradients computed from the training labeled data.
4. After training, A and B can be merged into W to create a fine-tuned model with no change in inference speed.
5. The fine-tuned model can be used for inference on the downstream task. Switching tasks just requires swapping in different A and B (trained on different labeled data).

## Empirical Results
- Datasets:
  - GLUE benchmark: Used for testing RoBERTa and DeBERTa.
  - E2E NLG Challenge: Used for testing GPT-2.
  - WikiSQL and SAMSum: Used for large-scale experiments on GPT-3.
    
- Model Architectures:
  - RoBERTa (base and large): The performance was evaluated on the GLUE benchmark.
  - DeBERTa: The performance was evaluated on the GLUE benchmark.
  - GPT-2 (medium and large): The performance was evaluated on the E2E NLG Challenge.
  - GPT-3 175B: The performance was evaluated on WikiSQL (natural language to SQL queries) and SAMSum (conversation summarization).

- Main Results:
  - On GLUE, LoRA matched or exceeded fine-tuning performance using 700x fewer parameters for RoBERTa Large.
  - In the E2E NLG Challenge, LoRA outperformed several baselines with comparable or fewer trainable parameters.
  - LoRA adaptation method achieved the highest accuracy compared to other methods on GPT-3, outperforming the Fine-Tuning method by 0.2% on multiple metrics.
  
- Efficiency:
  - Compared to fine-tuning GPT-3 175B, LoRA reduced parameter count by 10,000x and memory usage by 3x.
  - LoRA resulted in a 25% speedup in training throughput compared to full fine-tuning.
  - No inference speed slowdown compared to fine-tuning, unlike adapter methods.
 
tl;dr:
- The paper details various experiments conducted to demonstrate LoRA's effectiveness. LoRA achieves competitive performance compared to full fine-tuning across different NLP tasks while drastically reducing the number of trainable parameters.

**Discussion Question: Are there specific applications or domains where you think LoRA might be less suitable, and why?**

# LoRA Finetuning Pseudocode (Architecture Overview):
* Focusing only on matrices W^Q and W^K

```
Input: 
  - LLM model with parameters θ
  - Training data D = {(x_i, y_i)}
  - Learning rate η
  - Rank of approximation k
  - Batch size B
  - Number of epochs E
  
Output:
  - Fine-tuned model with parameters θ'

1: Initialize low-rank matrices A and B for each attention layer with rank k
2: Freeze the original weight matrices W^Q and W^K in the attention layers
3: for epoch = 1, 2, ..., E do
4:   for each batch (x, y) in D with size B do
5:     Compute gradients ∇U and ∇V w.r.t. the loss L(θ, (x, y))
6:     Update low-rank matrices U and V:
7:       A = A - η∇A
8:       B = B - η∇B
9:   end for
10: end for
11: Update weight matrices W^Q and W^K using A and B:
12:   W^Q = AB^T
13:   W^K = AB^T
14: Update the model parameters θ' with the updated weight matrices
15: Return fine-tuned model parameters θ'
```

# Critical Analysis
**Advantages:**
- Drastically reduces number of trainable parameters compared to full fine-tuning. This enables fine-tuning huge models like GPT-3 with limited compute.
- Avoids changing the original model weights, which could degrade the general knowledge acquired during pre-training.
- Allows efficient switching between tasks by just swapping the low-rank matrices.
- No inference speed penalty compared to full fine-tuning, as LoRA matrices can be merged into original weights.
- Achieves competitive performance to full fine-tuning on many NLP tasks.
- Provides insights into the low-rank structure of optimal model updates.

**Limitations:**
- May not reach full fine-tuning performance on some difficult tasks, as model capacity is constrained.
- Need to select which weight matrices to apply LoRA to. No clear optimal strategy known yet, although the paper does cover some empirical analysis of this.
- Restricts model updates to lie in a low-rank subspace, which could hypothetically limit expressiveness.
- LoRA rank is a new hyperparameter to tune, in addition to learning rate etc.

# Code Demonstration
Please refer to the `LoRA_guide.ipynb` notebook.

# Resources
- Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). Intrinsic dimensionality explains the effectiveness of language model fine-tuning. arXiv preprint arXiv:2012.13255. (https://arxiv.org/abs/2012.13255)
- Alexiuk, C. (2023, April 30). Low-rank adaption of large language models: Explaining the key concepts behind LoRA [Video]. YouTube. Retrieved from https://www.youtube.com/watch?v=dA-NhCtrrVE
- Niederfahrenhorst, A., Hakhamaneshi, K., & Ahmad, R. (2023, September 6). Fine-tuning LLMs: LoRA or Full-Parameter? An in-depth analysis with Llama 2. Anyscale Blog. Retrieved from https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2
- Roth, W. (2023, June 1). LoRA - Low-rank adaptation of AI large language models: LoRA and QLoRA explained simply [Video]. YouTube. https://www.youtube.com/watch?v=lixMONUAjfs
- Sooriyarachchi, A. (2023, August 30). Efficient fine-tuning with LoRA: A guide to optimal parameter selection for large language models. Databricks Engineering Blog. Retrieved from https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
