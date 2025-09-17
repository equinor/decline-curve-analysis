import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from dca.decline_curve_analysis import CurveLoss, Exponential

# Create some fake data
np.random.seed(12)
x = np.linspace(0, 15, num=2**5)
y = 10 * np.exp(-x * 0.2) + np.random.normal(size=len(x))
idx_outliers = [3, 8, 12, 18, 24, 30]
y[idx_outliers] = y[idx_outliers] * 2 + 10


# Setup plot
plt.figure(figsize=(7, 3))
plt.scatter(x, y, label="Data", color="black")
x_smooth = np.linspace(np.min(x), np.max(x))


# ===================================================================
def curve_func(t, theta1, theta2):
    """Curve function to fit to."""
    return Exponential(theta1, theta2).eval(t)


for p, metric in zip([1, 2], ["MAE", "RMSE"]):
    # Create loss and optimize it
    loss = CurveLoss(curve_func=curve_func, p=p)
    result = minimize(loss, x0=[0, 0], method="Nelder-Mead", args=(x, y))
    assert result.success
    curve = Exponential(*result.x)

    plt.plot(
        x_smooth,
        curve.eval(x_smooth),
        label=f"Minimizing {metric}",
    )


plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
