import matplotlib.pyplot as plt
import numpy as np

from dca.decline_curve_analysis import Arps, Exponential

# Default parameters
theta1 = 5
theta2 = 1

t = np.linspace(0, 5, num=2**10)
exponential = Exponential(theta1, theta2)

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(8, 3.25))
ax.set_title("Exponential and Arps")

# Plot exponential
ax.plot(t, exponential(t), label=f"{exponential!r}")

# Plot arps
for theta3 in [-2, -1, 0, 1, 2, 3]:
    arps = Arps(theta1, theta2, theta3)
    ax.plot(t, arps(t), label=f"{arps!r}")


ax.grid(True, ls="--", zorder=0, alpha=0.33)
ax.legend()
plt.tight_layout()
plt.show()
