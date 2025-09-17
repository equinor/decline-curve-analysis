import matplotlib.pyplot as plt
import numpy as np
from dca.decline_curve_analysis import Exponential
from dca.models import AR1Model


rng = np.random.default_rng(42)

# Use relatively few data points, for speed and for initial condition
n = 100
tau = np.ones(n)
t = np.cumsum(tau) - tau / 2

# Parameters for simulation
curve_params = (10, 0.02)
curve_func = Exponential.from_original_parametrization(*curve_params)
theta = curve_func.thetas
sigma = 0.1
p = 2

# Create a model and simulate from it
model = AR1Model(curve_cls=Exponential)


fig, axes = plt.subplots(1, 4, figsize=(8, 2))


for ax, phi in zip(axes.ravel(), [0, 0.66, 0.9, 0.99]):
    ax.set_title(f"$\phi$={phi}")

    model.update(theta=theta, sigma=sigma, phi=phi, p=p)
    y = model.simulate(tau=tau, t=t, seed=rng, simulations=3)

    ax.plot(t, curve_func.eval(t), color="black", zorder=9)
    ax.plot(t, y.T, zorder=5, alpha=0.5)

    ax.grid(True, ls="--", zorder=0, alpha=0.66)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


fig.tight_layout()
