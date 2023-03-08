# About

This repository contains boilerplate for PyTorch's training loop.

Some features included in this boilerplater are:

## 1. Periodic Checkpoints

- Automatically save checkpoints of model state, optimizer state and training scores every x epochs
- Load checkpoints

## 2. Gradient Scaling

- If the gradients are computed as `float16`, the value of gradients may underflow to 0.
- Gradients are scaled up after backpropagation and scaled down before optimizer step to solve this

## 3. Automatic Tensor Casting

- The outputs and loss of the model are lowered in precision for less memory usage.

## 4. Garbage Collection

- Every batch, the cache is cleared and non-used objects are flushed from memory.
