import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3]) - 1
y = np.array([100, 50, 25])

x_smooth = np.linspace(0, 3.5, num=2**10)
y_smooth = 100 * 2 ** -(x_smooth - 1)
y_smooth2 = 100 * 2 ** -(x_smooth - 0.5)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
# ==========================================
ax1.set_title(r"Shifting the curve")
ax1.bar(x, y, align="edge", width=1.0, color="none", edgecolor="black")

ax1.plot(x_smooth, y_smooth, label="Original fit")
ax1.plot(x_smooth, y_smooth2, label="Shifted fit")

ax1.set_xticks([0, 1, 2, 3])
ax1.grid(True, ls="--", alpha=0.4, zorder=0)
ax1.legend()

# ==========================================
ax2.set_title(r"Overestimation")
ax2.bar(x[:1], y[:1], align="edge", width=1.0, color="none", edgecolor="black")
x_smooth = np.linspace(0, 1, num=2**10)
ax2.plot(x_smooth, 100 - 100 * (x_smooth - 0.5), label="Linear function")
ax2.plot(
    x_smooth,
    100 - 100 * (x_smooth - 0.5) + 90 * (x_smooth - 0.5) ** 2,
    label="Convex decreasing function",
)
ax2.grid(True, ls="--", alpha=0.4, zorder=0)
ax2.legend()
ax2.set_xticks([0, 1])


fig.tight_layout()
plt.show()
