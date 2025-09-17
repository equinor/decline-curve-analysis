# NOTE: This example is quite large because it shows how to set up hyperparameter
# tuning from scratch using only the `dca` package. This can be avoided by
# using the `adca` package.

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from dca.datasets import load_monthly_sodir_production
from dca.decline_curve_analysis import Arps, CurveLoss


# Read data from a public source (SODIR - sokkeldirektoratet)
df = load_monthly_sodir_production()
FIELD = "STATFJORD"
df = df[df.prfInformationCarrier == FIELD]


# Process data
df = df.sort_values(["prfYear", "prfMonth"])
x = np.arange(len(df)) + 0.5  # Shift by 0.5 so integral matches production
y = df.prfPrdOilNetMillSm3.values
x = x / 12  # Convert to yearly resolution


# # Set up functions for hyperparameter tuning


def log_Arps(t, theta1, theta2, theta3):
    # Define this function to get the correct signature
    return Arps(theta1, theta2, theta3).eval_log(t)


def log_Arps_grad(t, theta1, theta2, theta3):
    # Define this function to get the correct signature
    return Arps(theta1, theta2, theta3).eval_grad_log(t)


def loss(parameters, x, y):
    """Return RMSE."""

    # Unpack params
    p, half_life, alpha = parameters

    # Used to split into train/validation sets
    rng = np.random.default_rng(42)

    # Create loss
    loss_function = CurveLoss(
        curve_func=log_Arps,
        curve_func_grad=log_Arps_grad,
        p=p,
        half_life=half_life,
        mu=MU_PRIOR,
        precision_matrix=alpha * np.eye(3),
    )

    residuals = []

    # Create random splits.
    # If the data has length 100, on average
    # this will cross validate a prediction
    # of length 50. Might not be exactly what
    # you want to do, but it's a rough estimate
    # how a long-term forecast error
    for random_split in range(9):
        i = rng.integers(0, len(x))

        # Split
        x_train, x_val = x[:i], x[i:]
        y_train, y_val = y[:i], y[i:]

        # Optimize: fit log model to log-data
        args = (x_train, np.log(y_train))
        optimization_result = sp.optimize.minimize(
            loss_function,
            x0=MU_PRIOR,
            method="BFGS",
            args=args,
            jac=loss_function.grad,
        )

        y_preds = log_Arps(x_val, *optimization_result.x)

        # Here we compute the residuals in the log space
        # Alternatively, they could be computed in the original space
        residuals.extend((y_preds - np.log(y_val)))

    return np.sqrt(np.mean(np.array(residuals) ** 2))


# Split
i = -10 * 12  # A ten year test period
x_train, x_test = x[:i], x[i:]
y_train, y_test = y[:i], y[i:]

# Set prior mean
# If we had more fields, we might've used a pilot estimate here
MU_PRIOR = np.array([10, 0, 0])  # Very rough guess


# Optimize hyperparameters


def transform_params(parameters):
    """Transform parameters from uniform sample space to sensible seach space."""
    p, half_life, alpha = parameters

    # Transform parameters
    p = 1 + sp.special.expit(p)  # Transform to [1, 2]
    half_life = 10**half_life
    alpha = 10**alpha

    return (p, half_life, alpha)


def inverse_transform_params(parameters):
    """Inverse transform."""
    p, half_life, alpha = parameters

    # Transform parameters
    p = sp.special.logit(p - 1)
    half_life = np.log10(half_life)
    alpha = np.log10(alpha)

    return (p, half_life, alpha)


def loss_param_transform(parameters, x, y):
    # Transform
    parameters_transformed = transform_params(parameters)
    # Evaluate, print and return
    return loss(parameters_transformed, x, y)


# The DIRECT algorithm does a good job at hyperparameter tuning
result = sp.optimize.direct(
    func=loss_param_transform,
    bounds=[(-4, 4), (-1, 1), (-1, 1)],
    args=(
        x_train,
        y_train,
    ),
    callback=lambda x: print(transform_params(x)),
    maxiter=1,
)

optimal_parameters = transform_params(result.x)

# Hardcoded parameters
optimal_parameters = (1.9593202223724138, 0.8431909292866258, 1.9783188827841642)
print(f"Optimal parameters {optimal_parameters}")


# # Fit to training data used optimized hyperparameters
opt_p, opt_half_life, opt_alpha = optimal_parameters

# Create loss
loss_function = CurveLoss(
    curve_func=log_Arps,
    curve_func_grad=log_Arps_grad,
    p=opt_p,
    half_life=opt_half_life,
    mu=MU_PRIOR,
    precision_matrix=opt_alpha * np.eye(3),
)

# Optimize: fit log model to log data
args = (x_train, np.log(y_train))
optimization_result = sp.optimize.minimize(
    loss_function,
    x0=MU_PRIOR,
    method="BFGS",
    args=args,
    jac=loss_function.grad,
)

optimization_result

# Plot results
plt.figure(figsize=(7, 3))
plt.title(FIELD)

plt.scatter(x_train, np.log(y_train), s=3, label="Train")
plt.scatter(x_test, np.log(y_test), s=3, label="Test")

# Smooth grid to plot the curve with
x_smooth = np.linspace(20, 60, num=2**10)
y_smooth = log_Arps(x_smooth, *optimization_result.x)
plt.plot(x_smooth, y_smooth, color="black", label="Forecast", ls="--")

plt.ylabel("Oil production (log10)")
plt.xlabel("Time (years)")
plt.legend()
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"forecast_{FIELD}.png", dpi=200)
plt.show()
