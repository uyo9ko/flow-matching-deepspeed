import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import MSELoss
from typing import *
from torchdiffeq import odeint

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

def exists(v):
    return v is not None

def identity(t):
    return t

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

        def get_noised_and_flows(t):
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

        flow, pred_flow, pred_data = get_noised_and_flows( padded_times)
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
