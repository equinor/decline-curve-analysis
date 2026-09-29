import matplotlib.pyplot as plt
import numpy as np

from dca.decline_curve_analysis import Arps

# Default parameters
q_1 = 10
h = 0.5
D = 1

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 3.25))
ax1, ax2, ax3 = axes.ravel()
t = np.linspace(0, 5)

# ==============================================================
for delta_q_1 in [-6, -3, 0, 3, 6]:
    curve = Arps.from_original_parametrization(q_1 + delta_q_1, h, D)
    ax1.set_title(r"Changing $q$")
    if np.isclose(delta_q_1, 0):
        ax1.plot(t, curve.eval(t), color="black")
    else:
        ax1.plot(t, curve.eval(t), color="gray")

# ==============================================================
for delta_h in [-0.45, -0.2, 0, 0.2, 0.45]:
    curve = Arps.from_original_parametrization(q_1, h + delta_h, D)
    ax2.set_title(r"Changing $h$")
    if np.isclose(delta_h, 0):
        ax2.plot(t, curve.eval(t), color="black")
    else:
        ax2.plot(t, curve.eval(t), color="gray")

# ==============================================================
for delta_D in [1 / 4, 1 / 2, 1, 2, 4]:
    curve = Arps.from_original_parametrization(q_1, h, D * delta_D)
    ax3.set_title(r"Changing $D$")
    if np.isclose(delta_D, 1):
        ax3.plot(t, curve.eval(t), color="black")
    else:
        ax3.plot(t, curve.eval(t), color="gray")


plt.tight_layout()
plt.show()
