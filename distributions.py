from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.distributions as D
from functorch import vmap, jacrev
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Density(ABC):
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, dim)
        Returns:
            log_density: (batch_size, 1)
        """
        pass

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, dim)
        Returns:
            score: (batch_size, dim)
        """
        x = x.unsqueeze(1)
        score = vmap(jacrev(self.log_density))(x)
        return score.squeeze((1, 2, 3))


class Sampleable(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Returns:
            samples: (num_samples, dim)
        """
        pass


class Gaussian(torch.nn.Module, Sampleable, Density):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)


class GaussianMixture(torch.nn.Module, Sampleable, Density):
    def __init__(self, means: torch.Tensor, covs: torch.Tensor, weights: torch.Tensor):
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means, covariance_matrix=self.covs, validate_args=False,
            ),
            validate_args=False,
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(cls, nmodes: int, std: float, scale: float = 10.0, seed: float = 0.0) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(cls, nmodes: int, std: float, scale: float = 10.0) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)


# ── Plotting utilities ────────────────────────────────────────────────────────

def hist2d_sampleable(sampleable: Sampleable, num_samples: int,
                      ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples)
    ax.hist2d(samples[:, 0].cpu(), samples[:, 1].cpu(), **kwargs)


def scatter_sampleable(sampleable: Sampleable, num_samples: int,
                       ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples)
    ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), **kwargs)


def imshow_density(density: Density, bins: int, scale: float,
                   ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    d = density.log_density(xy).reshape(bins, bins).T
    ax.imshow(d.cpu(), extent=[-scale, scale, -scale, scale], origin='lower', **kwargs)


def contour_density(density: Density, bins: int, scale: float,
                    ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    d = density.log_density(xy).reshape(bins, bins).T
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    ax.contour(x_np, y_np, d.cpu().numpy(), **kwargs)


if __name__ == '__main__':
    densities = {
        "Gaussian": Gaussian(mean=torch.zeros(2), cov=10 * torch.eye(2)).to(device),
        "Random Mixture": GaussianMixture.random_2D(nmodes=5, std=1.0, scale=20.0, seed=3.0).to(device),
        "Symmetric Mixture": GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=8.0).to(device),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (name, density) in enumerate(densities.items()):
        ax = axes[idx]
        ax.set_title(name)
        imshow_density(density, 100, 15, ax, vmin=-15, cmap=plt.get_cmap('Blues'))
        contour_density(density, 100, 15, ax, colors='grey', linestyles='solid', alpha=0.25, levels=20)
    plt.show()
