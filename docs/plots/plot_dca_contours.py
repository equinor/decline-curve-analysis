import matplotlib.pyplot as plt
import numpy as np

from dca.datasets import load_monthly_sodir_production
from dca import Arps, CurveLoss
from scipy.optimize import minimize
import functools
import itertools

# Load data to plot
df = (
    load_monthly_sodir_production()
    .loc[lambda df: df["prfInformationCarrier"] == "STATFJORD", :]
    .groupby(["prfYear", "prfMonth"])["prfPrdOilNetMillSm3"]
    .sum()
    .sort_index()
    .reset_index(drop=True)
)


x = df.index - df.index.min() + 0.5
x = x / 12
y = np.log(df.values)  # Take log of production


# Create a loss function
def log_Arps(t, theta1, theta2, theta3):
    # Define this function to get the correct signature
    return Arps(theta1, theta2, theta3).eval_log(t)


def log_Arps_grad(t, theta1, theta2, theta3):
    # Define this function to get the correct signature
    return Arps(theta1, theta2, theta3).eval_grad_log(t)


# Create loss
MU_PRIOR = np.array([0, 0, 0])
alpha = 0.1
loss = CurveLoss(
    curve_func=log_Arps,
    curve_func_grad=log_Arps_grad,
    p=1.5,
    half_life=25,
    mu=MU_PRIOR,
    precision_matrix=alpha * np.eye(3),
)
loss = functools.partial(loss, x=x, y=y)


# Optimize it
result = minimize(loss, x0=MU_PRIOR, method="BFGS")
# assert result.success
optimal_params = result.x.copy()
print(f"Optimal params: {optimal_params}")


# Create figure layout
fig = plt.figure(figsize=(8, 5))
spec = fig.add_gridspec(ncols=3, nrows=2)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[0, 2])
ax_bottom = fig.add_subplot(spec[1, :])
axes = iter([ax1, ax2, ax3])

# Plot the data
ax_bottom.scatter(x, y, s=4)
model = Arps(*optimal_params)
t_smooth = np.linspace(0, 55)
ax_bottom.plot(t_smooth, model.eval_log(t_smooth), color="black")
ax_bottom.set_xlabel("Time [years]")
ax_bottom.set_ylabel("log (production)")


variable_ranges = [param_i + np.linspace(-3, 3, num=2**5) for param_i in optimal_params]

# Loop through each fixed dimension
dimensions = [0, 1, 2]
for i, j in reversed(list(itertools.combinations(dimensions, 2))):
    ax = next(axes)
    k = (set(dimensions) - set([i, j])).pop()
    print(f"Fixed dimension: {k}. Free dimensions: {i, j}")

    # Mesh grid for plotting
    grid_i, grid_j = np.meshgrid(variable_ranges[i], variable_ranges[j])

    # Compute z values
    z_values = np.empty_like(grid_i)

    # Iterate over each grid point
    for index_i in range(grid_i.shape[0]):
        for index_j in range(grid_i.shape[1]):
            # Create a parameter vector with the fixed parameter set to its optimal value
            params = optimal_params.copy()
            params[i] = grid_i[index_i, index_j]
            params[j] = grid_j[index_i, index_j]
            assert np.isclose(params[k], optimal_params[k])
            # Evaluate the loss function at this parameter vector
            z_values[index_i, index_j] = np.log(loss(params))

    cp = ax.contour(grid_i, grid_j, z_values, levels=10)
    ax.clabel(cp, inline=True, fontsize=8)

    ax.set_xlabel(r"$\theta_" + str(i + 1) + "$")
    ax.set_ylabel(r"$\theta_" + str(j + 1) + "$")
    ax.set_title(r"Fixed param: $\theta_" + str(k + 1) + "$")

    ax.scatter(
        [optimal_params[i]],
        [optimal_params[j]],
        marker="x",
        color="red",
        s=100,
        zorder=9,
    )


fig.tight_layout()
