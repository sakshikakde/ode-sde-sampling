# SDE / ODE Sampling Dynamics

Implementations of two sampling algorithms - **Langevin dynamics** (SDE) and **flow matching** (ODE) - with animated visualizations of sample transport in 2D.

## Project structure

```
sde_ode/
├── distributions.py      # Base classes and 2D distributions
├── dynamics.py           # ODE/SDE abstractions and Euler simulators
├── langevin.py           # Langevin SDE + animation
├── flow_matching.py      # Flow matching ODE + animation
└── outputs/              # Saved videos (.mp4) and figures (.png)
```

# ODE vs SDE - Side by Side Comparison

## Dynamics

| | ODE | SDE |
|---|---|---|
| **Animation** | <video src="outputs/flow_matching_animation.mp4" width="480" autoplay loop muted></video> | <video src="outputs/langevin_animation.mp4" width="480" autoplay loop muted></video> |
| **Trajectories** | <img src="outputs/flow_matching_animation_full_trajectories.png" width="480"> | <img src="outputs/langevin_animation_full_trajectories.png" width="480"> |
| **Equation** | $dx_t = u_t(x)\,dt$ | $dx_t = u_t(x)\,dt + dW_t$ |
| **Noise** | None - deterministic | $dW_t$ - Brownian motion |
| **Path shape** | Smooth, straight | Curved, stochastic zigzag |

---

## Continuity Equation

| | ODE | SDE |
|---|---|---|
| **Equation** | $\dfrac{\partial p_t}{\partial t} = -\nabla \cdot (p_t\, u_t)$ | $\dfrac{\partial p_t}{\partial t} = -\nabla \cdot (p_t\, u_t) + \dfrac{1}{2}\nabla^2 p_t$ |
| **Extra term** | None | $\frac{1}{2}\nabla^2 p_t$ - diffusion spreads probability outward |
| **Meaning** | Probability is purely transported | Probability is transported **and** diffused |

---

## Equilibrium

| | ODE | SDE |
|---|---|---|
| **Condition** | $\nabla \cdot (p_t\, u_t) = 0$ | $\dfrac{\partial p^*}{\partial t} = 0$ |
| **Meaning** | Incompressible flow - flux has zero divergence | Distribution stops changing entirely |
| **Unique solution?** | No - many $u_t$ satisfy this | Yes - unique $u_t$ balances drift and diffusion |
| **Issue** | Multiple answers, no convergence guarantee | Converges to $p^*$ as $t \to \infty$ |

---

## Deriving $u_t$

### ODE - we design $u_t$ explicitly

Since equilibrium gives no unique answer, we **choose** a path directly.

**OT path:** pair $x_0 \sim p_0$ with $x_1 \sim p_1$, interpolate:

$$x_t = (1-t)\,x_0 + t\,x_1$$

Differentiate:

$$\boxed{u_t = x_1 - x_0}$$

Constant velocity, straight line, no ambiguity.

---

### SDE - $u_t$ is derived from equilibrium

Set $\partial p^* / \partial t = 0$ in the Fokker-Planck equation:

$$-\nabla \cdot (p_t\, u_t) + \frac{1}{2}\nabla^2 p_t = 0$$

Rearrange:

$$\nabla \cdot (p_t\, u_t) = \frac{1}{2}\nabla^2 p_t$$

Use the identity $\nabla p_t = p_t \nabla \log p_t$:

$$\frac{1}{2}\nabla^2 p_t
  = \frac{1}{2}\nabla \cdot (\nabla p_t)
  = \frac{1}{2}\nabla \cdot (p_t\, \nabla \log p_t)
  = \nabla \cdot \!\left(p_t \cdot \frac{1}{2}\nabla \log p_t\right)$$

Matching both sides:

$$\boxed{u_t = \frac{1}{2}\nabla \log p_t}$$

The drift must equal **half the score** of the target distribution.

---

## Sampling Algorithms

| | ODE | SDE |
|---|---|---|
| **Update rule** | $x_{t+dt} = x_t + u_t(x_t)\,dt$ | $x_{k+1} = x_k + \frac{\eta}{2}\nabla \log p^*(x_k) + \sqrt{\eta}\,\epsilon$ |
| **Noise at inference** | None | $\epsilon \sim \mathcal{N}(0,I)$ re-injected every step |
| **Example** | Flow Matching, DDIM | Langevin MCMC, DDPM |

---

## Key Insight

> **ODE:** No unique equilibrium - so we *design* $u_t$ (e.g. straight-line OT path).
> The continuity equation is satisfied by construction.
>
> **SDE:** The noise term $dW_t$ forces a unique stationary distribution $p^*$.
> The only drift that balances it is $u_t = \frac{1}{2}\nabla \log p^*$ - the score.

---

## Final Boxed Results

$$\textbf{ODE (Flow Matching):} \quad u_t = x_1 - x_0$$

$$\textbf{SDE (Langevin):} \quad u_t = \frac{1}{2}\nabla \log p_t$$