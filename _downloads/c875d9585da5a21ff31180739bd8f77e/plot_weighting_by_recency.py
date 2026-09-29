import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from dca.decline_curve_analysis import CurveLoss, Exponential

# Create some fake data
np.random.seed(12)
x = np.linspace(0, 20, num=2**5)
y = 10 * x * np.exp(-x * 0.2) + np.random.normal(size=len(x), scale=1)


# Setup plot
plt.figure(figsize=(7, 3))
plt.scatter(x, y, label="Data", color="black")
x_smooth = np.linspace(np.min(x), np.max(x))


# ===================================================================
def curve_func(t, theta1, theta2):
    """Curve function to fit to."""
    return Exponential(theta1, theta2).eval(t)


for half_life in [24, 6, 1]:
    # Create loss and optimize it
    loss = CurveLoss(curve_func=curve_func, p=2, half_life=half_life)
    result = minimize(loss, x0=[0, 0], method="Nelder-Mead", args=(x, y))
    assert result.success
    curve = Exponential(*result.x)

    plt.plot(
        x_smooth,
        curve.eval(x_smooth),
        label=f"{half_life=}",
    )


plt.grid(True)
plt.ylim([plt.ylim()[0], np.max(y) + 2])
plt.legend()
plt.tight_layout()
plt.show()
