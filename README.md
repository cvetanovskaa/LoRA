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
- LoRA is a technique that optimizes the fine-tuning process by selectively updating only a subset of parameters, specifically targeting the weight matrices in the attention layers, which are crucial for capturing long-range dependencies in the input data.
To achieve this, LoRA utilizes low-rank matrix decomposition, a technique that simplifies the weight matrices while preserving their essential characteristics.

- What is low-rank decomposition? A technique used to approximate a matrix with a product of two or more matrices that have lower dimensions compared to the original matrix.

<p float="left">
  <img src="https://github.com/cvetanovskaa/LoRA/assets/15224551/7846c998-c92e-4f47-9bb2-ded02045924f" width="500" />
  <img src="https://github.com/cvetanovskaa/LoRA/assets/15224551/ea70ad34-87b9-426b-b494-0f3266c40b70" width="500" />
</p>

## Practical Example:
- Transformer weights (according to "Attention is all you need" paper), have dimensions of d x k = 512 x 64 = 32,768 trainable parameters.
- In LoRA with rank 8, we have two matrices:
  - 512 x 8 = 4096 parameters
  - 8 x 64 = 512 parameters
  - Therefore, LoRA gives us 4608 trainable parameteres => 86% reduction

## LoRA Finetuning Pseudocode (Architecture Overview):
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

1: Initialize low-rank matrices U and V for each attention layer with rank k
2: Freeze the original weight matrices W^Q and W^K in the attention layers
3: for epoch = 1, 2, ..., E do
4:   for each batch (x, y) in D with size B do
5:     Compute gradients ∇U and ∇V w.r.t. the loss L(θ, (x, y))
6:     Update low-rank matrices U and V:
7:       U = U - η∇U
8:       V = V - η∇V
9:   end for
10: end for
11: Update weight matrices W^Q and W^K using U and V:
12:   W^Q = UV^T
13:   W^K = UV^T
14: Update the model parameters θ' with the updated weight matrices
15: Return fine-tuned model parameters θ'
```

# Critical Analysis
TBD

# Code Demonstration
Please refer to the `LoRA Guide.ipynb` notebook.

# Resources
- https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
- https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2
