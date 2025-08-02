# %%
from thermopack.cubic import cubic
import numpy as np
import matplotlib.pyplot as plt

# Initialization
eos = cubic("PR", "N2O,O2", "classic")

# Set NRTL parameters (example: tau_12 = tau_21 = 0.1, alpha = 0.2)
eos.set_nrtl_params(1, 2, 0.1, 0.1, 0.2)

# Optionally set kij or lij if needed
# eos.set_kij(1, 2, 0.01)

# Compute bubble points across compositions at fixed T
T = 200.0  # K
x1 = np.linspace(0.01, 0.99, 40)
P = []
y1 = []

for xi in x1:
    try:
        p_bub, y = eos.bubble_temperature(T, [xi, 1 - xi], phase="liquid")
        P.append(p_bub)
        y1.append(y[0])
    except Exception:
        P.append(np.nan)
        y1.append(np.nan)

plt.plot(x1, P, label="liquid x₁")
plt.plot(y1, P, "--", label="vapor y₁")
plt.xlabel("Mole fraction N₂O")
plt.ylabel("Pressure (Pa or bar)")
plt.legend()
plt.title("VLE: N₂O + O₂ at 200 K (PR + Wong–Sandler + NRTL)")
plt.grid()
plt.show()

# %%
