# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.image as mpimg
from scipy.optimize import minimize

import thermo_pvt as tp


kij = 0.011
lij = -0.061
figs = []

initial_temp = 20
cold_temp = -8
pi = 1520 / 14.50377  # bar gauge pressure
cold_usage = 0.40
warm_temp = 0


# %%
# Cole's 60:40 mis
entonox60 = tp.Mixture("N2O,O2", (0.6, 0.4))
entonox60.set_kij(1, 2, kij)
entonox60.set_lij(1, 2, lij)
env60 = entonox60.phase_envelope(t_min=233)

fig_pe, ax_pe = plt.subplots(1, 1, figsize=(8, 6), layout="tight")
levels = np.arange(0.25, 1, 0.05)  # or 0.1

env60.plot(
    fig=fig_pe,
    xlim=(cold_temp - 15, initial_temp + 5),
    ylim=(0, 150),
    split=False,
    lw=2,
    color="blue",
    composition=(tp.Phase.VAPOUR, "N2O"),
)
df_idt_60 = entonox60.isochoric_delta_T((1320 + 14.69) / 14.50377, cold_temp, 20, ax=ax_pe)
# %%
fig_pe, ax_pe = plt.subplots(1, 1, figsize=(8, 6), layout="tight")
levels = np.arange(0.25, 1, 0.05)  # or 0.1

cold_temp = -10
entonox50 = tp.Mixture("N2O,O2", (0.5, 0.5))
entonox50.set_kij(1, 2, kij)
entonox50.set_lij(1, 2, lij)
env50 = entonox50.phase_envelope(t_min=233)

env50.plot(
    fig=fig_pe,
    xlim=(cold_temp - 15, initial_temp + 5),
    ylim=(0, 150),
    split=False,
    lw=2,
    color="red",
    composition=(tp.Phase.VAPOUR, "N2O"),
)
df_idt_50 = entonox60.isochoric_delta_T((1320 + 14.69) / 14.50377, cold_temp, 20, ax=ax_pe)


# Initial flash conditions at cold temperature
p_cold = df_idt_50.index[0]
p_dew = df_idt_50[df_idt_50[("Fractions", "Vapor Fraction")] == 1].index[0]
t_dew = df_idt_50[df_idt_50[("Fractions", "Vapor Fraction")] == 1]["Temperature_C"].iloc[0]


df_cvd_50 = entonox50.constant_volume_depletion(
    initial_temp=cold_temp,
    initial_pressure=p_cold,
    initial_n_total=1.0,
    mol_fraction_to_remove=1.0,
)

p_dew2 = df_cvd_50[df_cvd_50[("Fractions", "Vapor Fraction")] < 1].index[-1]
z_dew2 = df_cvd_50[df_cvd_50[("Fractions", "Vapor Fraction")] == 1][("Vapor", "N2O")].iloc[0]
n_dew2 = df_cvd_50[df_cvd_50[("Fractions", "Vapor Fraction")] == 1][("Total", "N2O")].iloc[0]

points = {
    "A": (initial_temp, pi, 0.5, 0.5),
    "B": (t_dew, p_dew, 0.5, 0.5),
    "C": (cold_temp, p_cold, df_idt_50[("Vapor", "N2O")].iloc[-1], df_idt_50[("Total", "N2O")].iloc[-1]),
    "D": (cold_temp, p_dew2, z_dew2, n_dew2),
    "E": (cold_temp, 0, z_dew2, n_dew2),
}

# %%
# fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6), layout="tight")
# fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6), layout="tight")

points_to_use = {k: points[k] for k in ["A", "B", "C"]}

cooling_label = f"Cooling to {cold_temp:0.0f} °C"
disp_label = f"Dispensing at {cold_temp:0.0f} °C"

ff = []

for color in ["black", "grey"]:
    ff.append(
        original_env.plot(
            xlim=(cold_temp - 15, initial_temp + 5),
            split=False,
            label="Initial Phase Envelope",
            color=color,
            lw=1,
            cp_color=color,
            composition=(tp.Phase.VAPOUR, "N2O"),
        )
    )

    ax = ff[-1].axes[0]
    ax.set_ylim(0, 220)
    for text, (t, p, z, n) in points_to_use.items():
        ax.annotate(text, (t, p), xytext=(4, 4), textcoords="offset points", fontsize=14)

    ax.plot(
        np.array([point[0] for point in points_to_use.values()]),
        np.array([point[1] for point in points_to_use.values()]),
        marker="o",
        ls="--",
        label=cooling_label,
    )


fig1 = ff[0]
ax1 = ff[0].axes[0]
fig2 = ff[1]
ax2 = ff[1].axes[0]

env = entonox.phase_envelope(t_min=233)
env.plot(
    xlim=(cold_temp - 15, initial_temp + 5),
    split=False,
    label="Final Phase Envelope",
    color="darkblue",
    lw=1,
    cp_color="blue",
    fig=fig2,
    composition=(tp.Phase.VAPOUR, "N2O"),
)


points_to_use2 = {k: points[k] for k in ["C", "D", "E"]}

for text, (t, p, z, n) in points_to_use2.items():
    ax2.annotate(text, (t, p), xytext=(4, 4), textcoords="offset points", fontsize=14)

ax2.plot(
    np.array([point[0] for point in points_to_use2.values()]),
    np.array([point[1] for point in points_to_use2.values()]),
    marker="o",
    ls="--",
    label=disp_label,
)

ax1.set_title(f"50:50 N$_2$O/O$_2$ Mixture: Phase Envelope During Cooling to -20 °C")
ax2.set_title(f"50:50 N$_2$O/O$_2$ Mixture: Phase Envelope After Dispensing at -20 °C")
ax1.legend(fontsize=10, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.3))
ax2.legend(fontsize=10, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.3))


fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6), layout="tight")
ax3.set_xlim(200, 0)
df_idt_1.plot(y=("Vapor", "N2O"), label=f"Dispensed Gas Composition ({cooling_label})", ls="--", ax=ax3)
df_cvd_1.plot(y=("Vapor", "N2O"), label=f"Dispensed Gas Composition ({cooling_label})", ls="--", ax=ax3)
df_idt_1.plot(y=("Total", "N2O"), label=f"Cylinder Gas Composition ({cooling_label})", ls="-.", ax=ax3, color="C0")
df_cvd_1.plot(y=("Total", "N2O"), label=f"Cylinder Gas Composition ({cooling_label})", ls="-.", ax=ax3, color="C1")

ax3.plot(
    np.array([point[1] for point in points_to_use.values()]),
    np.array([point[2] for point in points_to_use.values()]),
    marker="o",
    lw=0,
    color="C0",
)

ax3.plot(
    np.array([point[1] for point in points_to_use2.values()]),
    np.array([point[3] for point in points_to_use2.values()]),
    marker="^",
    lw=0,
    color="C1",
)

ax3.plot(
    np.array([point[1] for point in points_to_use2.values()]),
    np.array([point[2] for point in points_to_use2.values()]),
    marker="o",
    lw=0,
    color="C1",
)

for text, (t, p, z, n) in points.items():
    ax3.annotate(text, (p, z), xytext=(4, 4), textcoords="offset points", fontsize=14)
    if z != n:
        ax3.annotate(f"{text}'", (p, n), xytext=(4, 4), textcoords="offset points", fontsize=14)

ax3.set_xlabel("Pressure [bara]")
ax3.set_ylabel("N$_2$O Concentration [dimensionless]")
ax3.set_ylim(0.2, 1)
ax3.legend(fontsize=10, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
ax3.set_title(f"Dispensed and Cylinder N$_2$O Concetration vs Pressure\n During Cooling to and Dispensing at -20 °C")


# %%
results = {}
max_n2o_gas = {}


temps = range(-40, 0, 5)
pressures = range(100, 250, 50)
for p_i in pressures:
    results[p_i] = {}
    max_n2o_gas[p_i] = []
    for cold_temp in temps:
        entonox = tp.Mixture("N2O,O2", (0.5, 0.5))
        entonox.set_kij(1, 2, kij)
        entonox.set_lij(1, 2, lij)

        df_cooling = entonox.isochoric_delta_T(p_i, 20, cold_temp)
        p_cold = df_cooling.index[-1]

        df_disp = entonox.constant_volume_depletion(
            initial_temp=cold_temp,
            initial_pressure=p_cold,
            initial_n_total=1.0,
            mol_fraction_to_remove=1.0,
        )

        results[p_i][cold_temp] = pd.concat([df_cooling, df_disp])
        max_n2o_gas[p_i].append(results[p_i][cold_temp]["Vapor"]["N2O"].max())
        print(f"{p_i:5.0f} bar {cold_temp:5.0f}°C {max_n2o_gas[p_i][-1]*100:4.1f}% N2O")


markers = ["o", "s", "D", "^", "v", "p", "*", "X"]
# %%
fig, ax = plt.subplots(figsize=(8, 6), layout="tight")
i = 0
for p_i in pressures:
    ax.plot(temps, np.array(max_n2o_gas[p_i]) * 100, marker=markers[i], mfc="white", label=f"Pi = {p_i:0.0f} bar")
    i += 1
ax.plot(temps, len(temps) * [80], ls="--", color="black", label="80% N2O Limit", marker="")
ax.set_xlabel("Cooling Temperature [°C]")
ax.set_ylabel("Maximum Gas N2O Concentration [%]")
ax.legend()
figs.append(fig)
# %%
results[200][-15][("Vapor", "N2O")].plot()
# %%
