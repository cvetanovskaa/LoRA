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
LoRA adapts the weight matrices of attention layers in LLMs by introducing low-rank trainable matrices, which significantly reduces the number of parameters to be trained during the fine-tuning process. This approach targets the attention mechanism of transformers, which is a critical component for capturing long-range dependencies in text.

## Low-Rank Decomposition
What is low-rank decomposition? A technique used to approximate a matrix with a product of two or more matrices that have lower dimensions compared to the original matrix. The key advantage of this decomposition is that it significantly reduces the number of trainable parameters.
<br>
- Hypothesis: low "intrisinc rank" i.e. the change in weights during model adaptation can be described almost as accurately using way less dimension than the models have.

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
LoRA's fine-tuning process involves three key steps:

1. Initialization of low-rank matrices for each attention layer. The dimensions are A ∈ R^{r x d}, B ∈ R^{d x r} where r is the rank and d is the input dimension.
2. Freezing the original weight matrices in attention layers so they do not get updated during fine-tuning.
3. Iteratively updating only the low-rank matrices (A and B) based on gradients computed from the training labeled data.
4. After training, A and B can be merged into W to create a fine-tuned model with no change in inference speed.
5. The fine-tuned model can be used for inference on the downstream task. Switching tasks just requires swapping in different A and B (trained on different labeled data).

## Experimental Results
The paper details various experiments conducted to demonstrate LoRA's effectiveness. LoRA achieves competitive performance compared to full fine-tuning across different NLP tasks while drastically reducing the number of trainable parameters.
  

# LoRA Finetuning Pseudocode (Architecture Overview):
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
- https://www.youtube.com/watch?v=lixMONUAjfs
- https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
- https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2
