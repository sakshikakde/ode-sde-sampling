import torch
import numpy as np
import seaborn as sns
from celluloid import Camera
from IPython.display import HTML
from matplotlib import pyplot as plt

from distributions import Density, Sampleable, Gaussian, GaussianMixture, imshow_density, device
from dynamics import ODE, every_nth_index


class FlowMatchingODE(ODE):
    """
    Per-particle constant-velocity ODE for flow matching / rectified flow.
    Each particle i travels along the straight line x0[i] → x1[i]:
        dx/dt = v[i] = x1[i] - x0[i]   (independent of x and t)
    """
    def __init__(self, v: torch.Tensor):
        self.v = v  # (num_samples, dim)

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.v


def animate_flow_matching(
    num_samples: int,
    source_distribution: Sampleable,
    target_distribution: Sampleable,
    density: Density,
    timesteps: torch.Tensor,
    animate_every: int,
    bins: int,
    scale: float,
    save_path: str = 'outputs/flow_matching_animation.mp4',
):
    """
    Animate flow matching dynamics.

    Samples x0 ~ source, x1 ~ target (randomly paired), then transports
    particles along straight lines xt = x0 + t*(x1-x0).  Velocity arrows
    show the constant per-particle vector x1-x0.  Also produces a static
    full-trajectory figure.
    """
    # Sample and randomly pair source / target
    x0 = source_distribution.sample(num_samples)                             # (N, 2)
    x1 = target_distribution.sample(num_samples)[torch.randperm(num_samples)]  # (N, 2)

    # Analytical trajectory: xt = x0 + t*(x1-x0)
    v = x1 - x0                                                              # (N, 2)
    xts = x0[:, None, :] + timesteps[None, :, None] * v[:, None, :]         # (N, T, 2)

    indices = every_nth_index(len(timesteps), animate_every)
    animate_timesteps = timesteps[indices]
    animate_xts = xts[:, indices]                                            # (N, frames, 2)

    n_traj = num_samples#min(100, num_samples)
    vx = v[:, 0].cpu().numpy()
    vy = v[:, 1].cpu().numpy()
    mag = np.sqrt(vx**2 + vy**2).clip(1e-8)

    import matplotlib
    _backend = matplotlib.get_backend()
    plt.switch_backend('Agg')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    camera = Camera(fig)
    for t_idx in range(len(animate_timesteps)):
        xt = animate_xts[:, t_idx]

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
    animation.save(save_path)
    plt.close()
    plt.switch_backend(_backend)

    # Static full-trajectory figure
    _, ax_traj = plt.subplots(figsize=(8, 8))
    imshow_density(density, bins, scale, ax_traj, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
    for i in range(n_traj):
        ax_traj.plot(xts[i, :, 0].cpu(), xts[i, :, 1].cpu(), color='black', alpha=0.3, linewidth=0.4)
    ax_traj.scatter(x0[:n_traj, 0].cpu(), x0[:n_traj, 1].cpu(),
                    color='green', s=15, zorder=3, label='start (x0)')
    ax_traj.scatter(x1[:n_traj, 0].cpu(), x1[:n_traj, 1].cpu(),
                    color='red', s=15, zorder=3, label='end (x1)')
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

    animate_flow_matching(
        num_samples=1000,
        source_distribution=source,
        target_distribution=target,
        density=target,
        timesteps=torch.linspace(0, 1.0, 1000).to(device),
        animate_every=100,
        bins=200,
        scale=15,
    )
