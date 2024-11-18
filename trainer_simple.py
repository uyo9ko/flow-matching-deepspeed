import argparse
import os
import re

import torch
import torch.nn as nn

from model import Unet
import datasets
    
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torch.nn import MSELoss 
from tqdm import tqdm
from typing import *
import numpy as np
import pandas as pd
import PIL
from timm import create_model
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor, Normalize


def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d 

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(torch.pi / 2 * t) + 1))

class RectifiedFlow(nn.Module):
    def __init__(self, net: nn.Module, device: torch.device) -> None:
        super().__init__()
        self.net = net
        self.device = device
        self.loss_fn = MSELoss()
        self.noise_schedule = cosmap

    def predict_flow(self, model, noised, *, times, eps = 1e-10):
        batch = noised.shape[0]
        # prepare maybe time conditioning for model
        model_kwargs = dict()
        times = rearrange(times, '... -> (...)')
        if times.numel() == 1:
            times = repeat(times, '1 -> b', b = batch)
        model_kwargs.update(**{'times': times})
        output = model(noised, **model_kwargs)
        return output

    def forward(self, data):
        noise = torch.randn_like(data)
        times = torch.rand(data.shape[0], device = self.device)
        padded_times = append_dims(times, data.ndim - 1)

        def get_noised_and_flows(model, t):
            # maybe noise schedule

            t = self.noise_schedule(t)

            # Algorithm 2 in paper
            # linear interpolation of noise with data using random times
            # x1 * t + x0 * (1 - t) - so from noise (time = 0) to data (time = 1.)

            noised = t * data + (1. - t) * noise

            # the model predicts the flow from the noised data

            flow = data - noise

            pred_flow = self.predict_flow(self.net, noised, times = t)

            # predicted data will be the noised xt + flow * (1. - t)
            pred_data = noised + pred_flow * (1. - t)

            return flow, pred_flow, pred_data

        flow, pred_flow, pred_data = get_noised_and_flows(self.net, padded_times)
        loss = self.loss_fn(flow, pred_flow)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        steps = 16,
        noise = None,
        data_shape: Tuple[int, ...] | None = None,
        **kwargs
    ):
        self.eval()

        def ode_fn(t, x):
            flow = self.predict_flow(self.net, x, times = t)
            return flow

        # start with random gaussian noise - y0
        noise = default(noise, torch.randn((batch_size, *data_shape), device = self.device))

        # time steps
        times = torch.linspace(0., 1., steps, device = self.device)

        # ode
        trajectory = odeint(ode_fn, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')

        sampled_data = trajectory[-1]

        self.train()

        return sampled_data


def collate_fn(batch):
    transform = Compose([
        Resize((256, 256)),  # Resize images to consistent dimensions
        ToTensor(),          # Convert PIL images to tensors and normalize to [0,1]
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Extract and process images from the batch
    images = [transform(item['jpeg'].convert('RGB')) for item in batch]
    return torch.stack(images)


def main():
    # Initialize model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(dim=64).to(device)
    flow_model = RectifiedFlow(net, device)
    
    # Training parameters
    num_epochs = 10
    learning_rate = 1e-4
    batch_size = 32
    
    # Optimizer
    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=learning_rate)
    
    data_set_path = 'imagenet-1k/data'
    dataset = datasets.load_dataset(path=data_set_path, split='train', streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Training loop
    for epoch in range(num_epochs):
        flow_model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                          total=1_281_167 // batch_size)  # Total images divided by batch size
        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = flow_model(batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            # Optional: Generate samples every N epochs
            if (step + 1) % 1000 == 0:
                with torch.no_grad():
                    samples = flow_model.sample(batch_size=4, data_shape=batch.shape[1:])
                    
                    # Create directory for samples if it doesn't exist
                    save_dir = os.path.join('samples', f'step_{step+1}')
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Convert and save each sample
                    for i, sample in enumerate(samples):
                        # Denormalize and convert to PIL image
                        sample = (sample.cpu().clamp(-1, 1) + 1) / 2 * 255
                        sample = sample.permute(1, 2, 0).numpy().astype(np.uint8)
                        img = PIL.Image.fromarray(sample)
                        
                        # Save the image
                        img.save(os.path.join(save_dir, f'sample_{i}.png'))
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        

                
        # Optional: Save checkpoint
        # if (epoch + 1) % 20 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': flow_model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': avg_loss,
        #     }, f'checkpoint_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    main()