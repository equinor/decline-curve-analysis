import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dca.decline_curve_analysis import Exponential
from dca.adca.well import Well

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Set up the true underlying model
model = Exponential.from_original_parametrization(C=10, k=0.33)

# Time on within each period
time_on = np.array([0.5, 0.2, 0.1, 0.8, 0.4, 0.3])

# Integrate the true model to determine production within each period
T1 = np.cumsum(time_on) - time_on
T2 = np.cumsum(time_on)
production = model.eval_integral_from_T1_to_T2(T1, T2)


fig, axes = plt.subplots(2, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = axes.ravel()

# ============== PLOT TRUE MODEL ==============
ax1.set_title("True model")
t_smooth = np.linspace(0, np.sum(time_on))
ax1.plot(t_smooth, model.eval(t_smooth))
ax1.set_xlabel("Time")

for i, cum_time_on_i in enumerate(np.cumsum(time_on)):
    ax1.scatter(
        [cum_time_on_i], [model.eval(cum_time_on_i)], s=25, color=COLORS[0], marker="|"
    )

# ============== PLOT RAW DATA ==============
ax2.set_title("Raw data")
ax2.scatter(
    np.arange(len(time_on)) + 1,
    production,
    color=COLORS[4],
    zorder=10,
    s=15,
    label="Observed production",
)

for i, time_on_i in enumerate(time_on):
    # Create a green line for "time_on" in each period
    start = i + (1 - time_on_i) / 2
    end = start + time_on_i
    ax2.plot([start, end], [0, 0], color=COLORS[2], lw=3)

    # Create a plot of the underlying model in each period
    region_x = np.linspace(start, end)
    region_t = np.linspace(T1[i], T2[i])
    region_y = model.eval(region_t)
    ax2.plot(region_x, region_y, color=COLORS[0])

    # Plot integral
    ax2.fill_between(
        region_x, np.zeros_like(region_y), region_y, color=COLORS[4], alpha=0.33
    )

ax2.set_xticks(range(len(time_on) + 1))
ax2.grid(True, ls="--", axis="x", alpha=0.5, zorder=0)
ax2.set_xlabel("Periods")
ax2.set_yticklabels([])

# ============== PLOT PRODUCING TIME DATA ==============
ax3.set_title("Producing time transform")
time = pd.period_range(start="2020-01-01", periods=len(production), freq="D")
well = Well(
    id=1,
    production=production,
    time=time,
    time_on=time_on,
    preprocessing="producing_time",
)

t_smooth = np.linspace(0, np.sum(time_on))
ax3.plot(t_smooth, model.eval(t_smooth), zorder=0)
x, y, w = well.get_curve_data()
ax3.scatter(x, y, s=5 + 25 * w, color=COLORS[4], zorder=10)
ax3.set_xlabel("Time")

# ============== PLOT CALENDAR TIMEDATA ==============
ax4.set_title("Calendar time transform")
time = pd.period_range(start="2020-01-01", periods=len(production), freq="D")
well = Well(
    id=1,
    production=production,
    time=time,
    time_on=time_on,
    preprocessing="calendar_time",
)

t_smooth = np.linspace(0, np.sum(time_on))
# ax4.plot(t_smooth, model.eval(t_smooth), zorder=0)
x, y, w = well.get_curve_data()
ax4.scatter(x, y, s=5 + 25 * w, color=COLORS[4], zorder=10)
ax4.set_xticks(range(len(time_on) + 1))
ax4.grid(True, ls="--", axis="x", alpha=0.5, zorder=0)
ax4.set_xlabel("Periods")
ax4.set_yticklabels([])
ax4.set_xticklabels([])

fig.tight_layout()
plt.show()
