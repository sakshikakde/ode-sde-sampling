from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xt: (batch_size, dim)
            t:  scalar
        Returns:
            drift: (batch_size, dim)
        """
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xt: (batch_size, dim)
            t:  scalar
        Returns:
            drift: (batch_size, dim)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xt: (batch_size, dim)
            t:  scalar
        Returns:
            diffusion: (batch_size, dim)
        """
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Single discretization step from t to t+dt."""
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Simulate from ts[0] to ts[-1], returning only the final state."""
        for t_idx in range(len(ts) - 1):
            x = self.step(x, ts[t_idx], ts[t_idx + 1] - ts[t_idx])
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Simulate and record every state.
        Returns:
            xs: (batch_size, num_timesteps, dim)
        """
        xs = [x.clone()]
        for t_idx in tqdm(range(len(ts) - 1)):
            x = self.step(x, ts[t_idx], ts[t_idx + 1] - ts[t_idx])
            xs.append(x.clone())
        return torch.stack(xs, dim=1)


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return xt + self.ode.drift_coefficient(xt, t) * dt


class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return (xt
                + self.sde.drift_coefficient(xt, t) * dt
                + self.sde.diffusion_coefficient(xt, t) * torch.sqrt(dt) * torch.randn_like(xt))


def every_nth_index(num_timesteps: int, n: int) -> torch.Tensor:
    """Indices to subsample a trajectory at every n-th step, always including the last."""
    if n == 1:
        return torch.arange(num_timesteps)
    return torch.cat([
        torch.arange(0, num_timesteps - 1, n),
        torch.tensor([num_timesteps - 1]),
    ])
