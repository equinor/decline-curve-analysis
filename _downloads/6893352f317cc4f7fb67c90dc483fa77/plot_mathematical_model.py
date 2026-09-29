import matplotlib.pyplot as plt
import numpy as np

from dca.decline_curve_analysis import Exponential

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Set up the true underlying model
model = Exponential.from_original_parametrization(C=10, k=0.33)

# Time on within each period
time_on = np.array([0.5, 0.2, 0.1, 0.8, 0.4, 0.3])

# Integrate the true model to determine production within each period
T1 = np.cumsum(time_on) - time_on
T2 = np.cumsum(time_on)
production = model.eval_integral_from_T1_to_T2(T1, T2)


fig, ax = plt.subplots(1, 1, figsize=(7, 4))

# ============== PLOT RAW DATA ==============
ax.set_title(
    "Model - we receive data at the end of each period\nthat is the integral of an unknown curve within that period"
)
ax.scatter(
    np.arange(len(time_on)) + 1,
    production,
    color=COLORS[4],
    zorder=10,
    s=35,
    label="Observed production",
)

for i, time_on_i in enumerate(time_on):
    # Create a green line for "time_on" in each period
    start = i + (1 - time_on_i) / 2
    end = start + time_on_i
    if i == 0:
        ax.plot([start, end], [0, 0], color=COLORS[2], lw=3, label="time_on")
    else:
        ax.plot([start, end], [0, 0], color=COLORS[2], lw=3)

    # Create a plot of the underlying model in each period
    region_x = np.linspace(start, end)
    region_t = np.linspace(T1[i], T2[i])
    region_y = model.eval(region_t)
    if i == 0:
        ax.plot(region_x, region_y, color=COLORS[0], label="Decline curve")
    else:
        ax.plot(region_x, region_y, color=COLORS[0])

    # Plot integral
    ax.fill_between(
        region_x, np.zeros_like(region_y), region_y, color=COLORS[4], alpha=0.33
    )

ax.set_xticks(range(len(time_on) + 1))
ax.grid(True, ls="--", axis="x", alpha=0.5, zorder=0)
ax.set_xlabel("Periods")
ax.set_yticklabels([])
ax.legend()

fig.tight_layout()
plt.show()
