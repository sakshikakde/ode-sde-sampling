import torch
import numpy as np
import seaborn as sns
from celluloid import Camera
from IPython.display import HTML
from matplotlib import pyplot as plt
import os

from distributions import Density, Sampleable, Gaussian, GaussianMixture, imshow_density, device
from dynamics import SDE, Simulator, EulerMaruyamaSimulator, every_nth_index


class LangevinSDE(SDE):
    """
    Overdamped Langevin SDE: dx = (σ²/2) ∇log p(x) dt + σ dW
    Stationary distribution is p(x).
    """
    def __init__(self, sigma: float, density: Density):
        self.sigma = sigma
        self.density = density

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.sigma ** 2 * self.density.score(xt)

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.ones_like(xt)


def graph_dynamics(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator,
    density: Density,
    timesteps: torch.Tensor,
    plot_every: int,
    bins: int,
    scale: float,
    save_path: str = 'outputs/langevin_steps.png',
):
    """Static grid of snapshots showing sample evolution over time."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x0 = source_distribution.sample(num_samples)
    xts = simulator.simulate_with_trajectory(x0, timesteps)
    indices = every_nth_index(len(timesteps), plot_every)
    plot_timesteps = timesteps[indices]
    plot_xts = xts[:, indices]

    fig, axes = plt.subplots(2, len(plot_timesteps), figsize=(8 * len(plot_timesteps), 16))
    axes = axes.reshape((2, len(plot_timesteps)))
    for t_idx in range(len(plot_timesteps)):
        t = plot_timesteps[t_idx].item()
        xt = plot_xts[:, t_idx]

        scatter_ax = axes[0, t_idx]
        imshow_density(density, bins, scale, scatter_ax, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
        scatter_ax.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), marker='x', color='black', alpha=0.75, s=15)
        scatter_ax.set_title(f'Samples at t={t:.1f}', fontsize=15)
        scatter_ax.set_xticks([])
        scatter_ax.set_yticks([])

        kde_ax = axes[1, t_idx]
        imshow_density(density, bins, scale, kde_ax, vmin=-15, alpha=0.5, cmap=plt.get_cmap('Blues'))
        sns.kdeplot(x=xt[:, 0].cpu(), y=xt[:, 1].cpu(), alpha=0.5, ax=kde_ax, color='grey')
        kde_ax.set_title(f'Density at t={t:.1f}', fontsize=15)
        kde_ax.set_xticks([])
        kde_ax.set_yticks([])
        kde_ax.set_xlabel("")
        kde_ax.set_ylabel("")
    plt.savefig(save_path)
    plt.show()


def animate_dynamics(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator,
    density: Density,
    timesteps: torch.Tensor,
    animate_every: int,
    bins: int,
    scale: float,
    save_path: str = 'outputs/langevin_animation.mp4',
):
    """
    Animated video of sample evolution with velocity quivers and trajectory trails.
    Also produces a static full-trajectory figure.
    """
    x0 = source_distribution.sample(num_samples)
    xts = simulator.simulate_with_trajectory(x0, timesteps)          # (N, T, 2)
    indices = every_nth_index(len(timesteps), animate_every)
    animate_timesteps = timesteps[indices]
    animate_xts = xts[:, indices]                                     # (N, frames, 2)

    dynamics = simulator.sde if hasattr(simulator, 'sde') else simulator.ode
    n_traj = num_samples #min(100, num_samples)

    import matplotlib
    _backend = matplotlib.get_backend()
    plt.switch_backend('Agg')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    camera = Camera(fig)
    for t_idx in range(len(animate_timesteps)):
        t_tensor = animate_timesteps[t_idx]
        xt = animate_xts[:, t_idx]

        with torch.no_grad():
            velocity = dynamics.drift_coefficient(xt, t_tensor)      # (N, 2)
        vx = velocity[:, 0].cpu().numpy()
        vy = velocity[:, 1].cpu().numpy()
        mag = np.sqrt(vx**2 + vy**2).clip(1e-8)

        # Panel 1: scatter + velocity quivers
        scatter_ax = axes[0]
        imshow_density(density, bins, scale, scatter_ax, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
        scatter_ax.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), marker='x', color='black', alpha=0.75, s=15)
        scatter_ax.quiver(
            xt[:, 0].cpu(), xt[:, 1].cpu(),
            vx / mag, vy / mag, mag,
            cmap='RdYlGn', alpha=0.7, scale=40, width=0.003,
        )
        scatter_ax.set_title('Samples')

        # Panel 2: KDE
        kde_ax = axes[1]
        imshow_density(density, bins, scale, kde_ax, vmin=-15, alpha=0.5, cmap=plt.get_cmap('Blues'))
        sns.kdeplot(x=xt[:, 0].cpu(), y=xt[:, 1].cpu(), alpha=0.5, ax=kde_ax, color='grey')
        kde_ax.set_title('Density of Samples', fontsize=15)
        kde_ax.set_xticks([])
        kde_ax.set_yticks([])
        kde_ax.set_xlabel("")
        kde_ax.set_ylabel("")

        # Panel 3: growing trajectory lines
        traj_ax = axes[2]
        imshow_density(density, bins, scale, traj_ax, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
        for i in range(n_traj):
            path = animate_xts[i, :t_idx + 1]
            traj_ax.plot(path[:, 0].cpu(), path[:, 1].cpu(), color='black', alpha=0.3, linewidth=0.5)
        traj_ax.scatter(animate_xts[:n_traj, t_idx, 0].cpu(), animate_xts[:n_traj, t_idx, 1].cpu(),
                        color='red', s=10, zorder=3, alpha=0.6)
        traj_ax.set_title('Trajectories')
        traj_ax.set_xticks([])
        traj_ax.set_yticks([])

        camera.snap()

    animation = camera.animate()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    animation.save(save_path)
    plt.close()
    plt.switch_backend(_backend)

    # Static full-trajectory figure
    _, ax_traj = plt.subplots(figsize=(8, 8))
    imshow_density(density, bins, scale, ax_traj, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
    for i in range(n_traj):
        ax_traj.plot(xts[i, :, 0].cpu(), xts[i, :, 1].cpu(), color='black', alpha=0.3, linewidth=0.4)
    ax_traj.scatter(xts[:n_traj, 0, 0].cpu(), xts[:n_traj, 0, 1].cpu(),
                    color='green', s=15, zorder=3, label='start')
    ax_traj.scatter(xts[:n_traj, -1, 0].cpu(), xts[:n_traj, -1, 1].cpu(),
                    color='red', s=15, zorder=3, label='end')
    ax_traj.legend()
    ax_traj.set_title('Full Trajectories', fontsize=15)
    ax_traj.set_xticks([])
    ax_traj.set_yticks([])
    plt.savefig(save_path.replace('.mp4', '_full_trajectories.png'))
    plt.show()

    return HTML(animation.to_html5_video())


if __name__ == '__main__':
    target = GaussianMixture.random_2D(nmodes=5, std=0.75, scale=15.0, seed=3.0).to(device)
    source = Gaussian(mean=torch.zeros(2), cov=20 * torch.eye(2)).to(device)
    sde = LangevinSDE(sigma=0.6, density=target)
    simulator = EulerMaruyamaSimulator(sde)

    graph_dynamics(
        num_samples=1000,
        source_distribution=source,
        simulator=simulator,
        density=target,
        timesteps=torch.linspace(0, 5.0, 1000).to(device),
        plot_every=334,
        bins=200,
        scale=15,
    )

    animate_dynamics(
        num_samples=1000,
        source_distribution=source,
        simulator=simulator,
        density=target,
        timesteps=torch.linspace(0, 5.0, 1000).to(device),
        animate_every=100,
        bins=200,
        scale=15,
    )
