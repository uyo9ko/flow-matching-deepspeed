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

class VPDiffusionFlowMatching(nn.Module):
    def __init__(self, net: nn.Module, device: torch.device) -> None:
        super().__init__()
        self.net = net
        self.device = device
        self.beta_min = 0.1
        self.beta_max = 20.0
        self.eps = 1e-5
        self.loss_fn = MSELoss()
        self.noise_schedule = cosmap

    def T(self, s: torch.Tensor) -> torch.Tensor:

        return self.beta_min * s + 0.5 * (s ** 2) * (self.beta_max - self.beta_min)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        
        return self.beta_min + t*(self.beta_max - self.beta_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:

        return torch.exp(-0.5 * self.T(t))

    def mu_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:

        return self.alpha(1. - t) * x_1

    def sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:

        return torch.sqrt(1. - self.alpha(1. - t) ** 2)

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:

        num = torch.exp(-self.T(1. - t)) * x - torch.exp(-0.5 * self.T(1.-t))* x_1
        denum = 1. - torch.exp(- self.T(1. - t))
        return - 0.5 * self.beta(1. - t) * (num/denum)

    def predict_flow(self, model, noised, *, times):
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

            # noised = t * data + (1. - t) * noise
            # noised = t * data + (1 - (1 - self.sig_min) * t) * noise
            noised = self.mu_t(t, data) + self.sigma_t(t, data) * noise

            # the model predicts the flow from the noised data

            # flow = data - noise
            # flow = data - (1 - self.sig_min) * noise
            flow = self.u_t(t, noised, data) 

            pred_flow = self.predict_flow(self.net, noised, times = t)

            # predicted data will be the noised xt + flow * (1. - t)
            pred_data = noised + pred_flow * (1. - t)

            return flow, pred_flow, pred_data

        flow, pred_flow, pred_data = get_noised_and_flows(padded_times)
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


def test():
    # Set device
    import sys
    sys.path.append('/data_training/larry/code/dit/flow-matching-deepspeed')
    from model import Unet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test parameters
    batch_size = 8
    data_dim = (3, 256, 256)
    
    # Create model
    net = Unet(dim=64).to(device)
    flow_model = VPDiffusionFlowMatching(net, device)
    flow_model.to(device)

    # Test forward pass
    print("\nTesting forward pass...")
    # Fix: Unpack data_dim when creating fake data
    fake_data = torch.randn(batch_size, *data_dim).to(device)
    loss = flow_model(fake_data)
    print(f"Forward pass loss: {loss.item()}")

    # Test sampling
    print("\nTesting sampling...")
    samples = flow_model.sample(
        batch_size=batch_size,
        data_shape=data_dim,  # This is fine as is
        steps=16
    )
    print(f"Generated samples shape: {samples.shape}")
    print(f"Samples mean: {samples.mean():.4f}")
    print(f"Samples std: {samples.std():.4f}")

if __name__ == "__main__":
    test()