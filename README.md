# Image generation trainer
A streamlined implementation of a flow-matching image generation task, powered by **Accelerate**.

## Overview  
This project adapts and simplifies code from the [Rectified Flow (PyTorch)](https://github.com/lucidrains/rectified-flow-pytorch) repository, bringing a lightweight and efficient setup for image generation using DeepSpeed.

## Features  
- Minimal setup for flow-matching-based image generation.  
- Efficient acceleration with **Accelerate** and **DeepSpeed**.  
- Simplified code structure for ease of customization and understanding.  
- Trains on the [ImageNet-1k dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k).  

## TODO  
- [x] Implement conditional flow-matching based on class labels.
- [x] Add image VAE encoding.
- [x] Add sit model.
- [ ] Add efficietn dit archtecture (sana).
- [ ] Change different optimizers like [Shampoo](https://github.com/facebookresearch/optimizers)
