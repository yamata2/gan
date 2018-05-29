# Generative Adversarial Network (GAN)

Generative Adversarial Network implemented by Tensorflow

Last updated: May. 30th, 2018.
Author: Tatsuro Yamada <<ymt2.casino@gmail.com>>

## Requirements
- Python 2.7
- Tensorflow 1.7
- NumPy 1.11

## Implementation
- GAN
  - normal loss  log(D(1-G(z)))
  - alternative loss -log(D(G(z)))
  - Wasserstein loss

## Example
```
$ cd train/examples/gan000
$ ./train.sh
```