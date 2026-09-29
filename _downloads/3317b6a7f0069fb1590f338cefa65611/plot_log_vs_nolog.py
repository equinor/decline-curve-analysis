import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def func(x, a, b):
    return a * np.exp(-x * b)


def func_logged(x, a, b):
    return a - b * x


# Generate some data
np.random.seed(6)
x = np.arange(25)
y = (
    10 * np.exp(-x * 0.2) * np.exp(np.random.randn(len(x)) / 10)
    + np.random.randn(len(x)) / 10
)
y = np.maximum(y, 0.01)


plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.title("Normal-space")
plt.scatter(x, y, label="Data")
x_smooth = np.linspace(np.min(x), np.max(x))


# ===================================================================
def loss_function(parameters, x, y):
    return np.sum(np.abs(func(x, *parameters) - y) ** 2)


def loss_function_logged(parameters, x, y):
    return np.sum(np.abs(func_logged(x, *parameters) - y) ** 2)


result = minimize(loss_function, x0=[0, 0], method="BFGS", args=(x, y))
plt.plot(
    x_smooth, func(x_smooth, *result.x), color="red", label="Fitted in normal-space"
)

result = minimize(loss_function_logged, x0=[0, 0], method="BFGS", args=(x, np.log(y)))
plt.plot(
    x_smooth,
    np.exp(func_logged(x_smooth, *result.x)),
    color="blue",
    label="Fitted in log-space",
)

plt.ylabel("y")
plt.legend()
plt.grid(True)


# ==============================================================================
plt.subplot(1, 2, 2)
plt.title("Log-space")
plt.scatter(x, np.log(y), label="Data")
x_smooth = np.linspace(np.min(x), np.max(x))

result = minimize(loss_function, x0=[0, 0], method="BFGS", args=(x, y))
plt.plot(
    x_smooth,
    np.log(func(x_smooth, *result.x)),
    color="red",
    label="Fitted in normal-space",
)

result = minimize(loss_function_logged, x0=[0, 0], method="BFGS", args=(x, np.log(y)))
plt.plot(
    x_smooth,
    (func_logged(x_smooth, *result.x)),
    color="blue",
    label="Fitted in log-space",
)

plt.ylabel("log y")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
