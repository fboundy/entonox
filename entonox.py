# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.image as mpimg
from scipy.optimize import minimize

import thermo_pvt as tp


# %%
def plot_lab_data(data, mixture=None, ax=None):
    comps = np.arange(0, 0.8, 0.005)
    markers = {
        "liquid": "^",
        "vapour": "o",
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    i = 0
    for temp, group in data.groupby("T_C"):
        color = f"C{i}"
        for phase in markers:
            marker = markers[phase]
            label = f"Observed {phase.title()} Data" if i == 0 else None  # <- omit label entirely after first

            filtered = group[group["phase"] == phase]
            ax.plot(
                filtered["Z_O2"],
                filtered["P_bar"],
                linestyle="None",
                marker=marker,
                markerfacecolor="white",
                markeredgecolor=color,
                label=label,
            )

        if mixture is not None:
            cp = []
            bpc = []
            dpc1 = []
            dpc2 = []
            zdp2 = []
            zdp1 = []

            calc_bp = True
            for o2 in comps:
                mixture.set_z((1 - o2, o2))

                try:
                    cp.append([o2] + list(mixture.critical()))
                except:
                    cp.append([o2] + [np.nan] * 3)

                try:
                    p = mixture.dew_pressures(temp + 273.15) / 1e5 / 1.01325
                    if ~np.isnan(p[0]):
                        dpc1.append(p[0])
                        zdp1.append(o2)

                    if ~np.isnan(p[1]):
                        dpc2.append(p[1])
                        zdp2.append(o2)

                except:
                    pass

                if calc_bp:
                    try:
                        bp = mixture.bubble_pressure(temp + 273.15)[0]
                        bpc.append(bp / 1e5 / 1.01325)

                        calc_bp = bp / cp[-1][3] < 0.98

                    except Exception as e:
                        bpc.append(np.nan)
                else:
                    bpc.append(np.nan)

            dpc2.reverse()
            zdp2.reverse()

            if i == 0:
                ax.plot(comps, np.array(cp)[:, 3] / 1e5, ls="--", label=f"Model Critical Point Locus", color="black")

            ax.plot(
                comps,
                bpc,
                label=f"Model at {temp}°C",
                color=color,
            )
            ax.plot(zdp1 + zdp2, dpc1 + dpc2, color=color)

        i += 1
    ax.set_xlabel("Concentration of O$_2$ [dimensionless]")
    ax.set_xlim(0.8, 0)
    ax.set_ylabel("Pressure [bar]")
    ax.set_ylim(0)
    ax.legend()
    ax.legend(fontsize=10, ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.3))

    return ax


def model_error(params, *args):
    kij, lij = params

    model = args[0]
    df = args[1]

    model.set_kij(1, 2, kij)
    model.set_lij(1, 2, lij)

    err = 0
    n = 0
    for row in df.iterrows():
        try:
            e = model.pressure_error(row[1]["T_K"], row[1]["Z_O2"], row[1]["P_Pa"])
            if e is not None:
                n += 1
                err += e / 1e5
        except:
            pass
    if n > 0:
        rmse = (err / n) ** 0.5
    else:
        raise ValueError("Unable to calculate model pressure error")
    return rmse


# %%
from bracken_data import df

img = mpimg.imread("Fig8.png")  # or your path

OPTIMISE = False
figs = []

initial_temp = 20
cold_temp = -20
pi = 3000 / 14.50377  # bar gauge pressure
cold_usage = 0.40
warm_temp = 0

# %%
entonox = tp.Mixture("N2O,O2", (0.5, 0.5), mixing="classic")

if OPTIMISE:
    bounds = [
        (0, 0.04),
        (-0.1, 0),
    ]

    res = minimize(
        model_error,
        (0, 0),
        args=(entonox, df),
        options={
            "disp": True,
            "gtol": 1e-3,
        },
        bounds=bounds,
        method="Powell",
    )
    kij = res[0]
    lij = res[1]
else:
    kij = 0.011
    lij = -0.061

# %%
fig_pe, ax_pe = plt.subplots(1, 1, figsize=(8, 6), layout="tight")
levels = np.arange(0.25, 0.55, 0.05)  # or 0.1

names = ["Un-tuned", "Tuned"]

for c, kl in enumerate([(0, 0), (kij, lij)]):
    color = f"C{c}"
    fig, ax = plt.subplots(figsize=(8, 6), layout="tight")
    entonox.set_kij(1, 2, kl[0])
    entonox.set_lij(1, 2, kl[1])

    plot_lab_data(df[df["T_C"].isin(range(-30, 10, 10))], mixture=entonox, ax=ax)
    ax.set_title(
        f"{names[c]} N$_2$O/O$_2$ Vapour Liquid Equilibrium: EOS Model vs Lab Data\n(kij={kl[0]:0.3f} lij={kl[1]:0.3f}) RMS Error = {model_error(kl, entonox, df):0.2f} bar"
    )
    figs.append(fig)

    entonox.set_z((0.5, 0.5))
    env = entonox.phase_envelope(t_min=cold_temp + 273.15 - 15)
    env.plot(
        fig=fig_pe,
        xlim=(cold_temp - 15, initial_temp + 5),
        ylim=(0, 150),
        split=False,
        label=f"{names[c]} Phase Envelope: kij={kij:0.3f} lij={lij:0.3f}",
        color=color,
        lw=2,
        cp_color=color,
        composition=(tp.Phase.VAPOUR, "N2O"),
    )


ax_pe.legend(fontsize=10, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
ax_pe.set_title(f"50:50 N$_2$O/O$_2$ Mixture: Tuned vs Un-tuned Phase Envelope")


# %%
fig = None

cp = []
for o2 in np.arange(0.0, 0.8, 0.1):
    mix = tp.Mixture("N2O,O2", (1 - o2, o2), mixing="classic")
    mix.set_kij(1, 2, kij)
    mix.set_lij(1, 2, lij)
    cp.append([o2] + list(mix.critical()))

res = pd.DataFrame(cp).set_axis(["O2", "Tc", "Vc", "Pc"], axis=1)

res["Tc"] -= 273.15
res["Pc"] /= 1e5
res.set_index("O2")

res.to_csv("Pc.csv")

data = np.array(
    (
        (-30, 32.23195357),
        (-25, 38.46527273),
        (-20, 45.58780881),
        (-15, 55.02507639),
        (-10, 66.52132159),
        (-10, 113.4668678),
    )
)

for o2 in np.arange(0.0, 0.8, 0.1):
    mix = tp.Mixture("N2O,O2", (1 - o2, o2), mixing="classic")
    mix.set_kij(1, 2, kij)
    mix.set_lij(1, 2, lij)

    env = mix.phase_envelope(step_size_factor=0.1)
    fig = env.plot(fig=fig, split=False, lw=3, cp=False, xlim=(-30, 37.5))

ax = fig.axes[0]
ax.plot(data[:, 0], data[:, 1], lw=0, marker="o", markersize=10, mfc="white")
res.plot(x="Tc", y="Pc", ax=ax, ls="--")


# Load the PNG image
img = mpimg.imread("Fig6.png")  # or your path


# Show the image as the background
extent = (-30, 37.5, 0, 140)
ax.imshow(img, extent=extent, aspect="auto")


plt.show()


# %%


original = tp.Mixture("N2O,O2", (0.5, 0.5))
original.set_kij(1, 2, kij)
original.set_lij(1, 2, lij)
original_env = original.phase_envelope(t_min=233)


entonox = tp.Mixture("N2O,O2", (0.5, 0.5))
entonox.set_kij(1, 2, kij)
entonox.set_lij(1, 2, lij)
env = entonox.phase_envelope(t_min=233)

levels = np.arange(0.25, 1, 0.05)  # or 0.1

pi_pa = tp.convert_pressure(pi, "bar", inverse=True)
df_idt_1 = entonox.isochoric_delta_T(pi, 20, cold_temp, ax=ax, plot_start=True, plot_end=False)

# Initial flash conditions at cold temperature
p_cold = df_idt_1.index[-1]
p_dew = df_idt_1[df_idt_1[("Fractions", "Vapor Fraction")] == 1].index[-1]
t_dew = df_idt_1[df_idt_1[("Fractions", "Vapor Fraction")] == 1]["Temperature_C"].iloc[-1]


df_cvd_1 = entonox.constant_volume_depletion(
    initial_temp=cold_temp,
    initial_pressure=p_cold,
    initial_n_total=1.0,
    mol_fraction_to_remove=1.0,
)

p_dew2 = df_cvd_1[df_cvd_1[("Fractions", "Vapor Fraction")] < 1].index[-1]
z_dew2 = df_cvd_1[df_cvd_1[("Fractions", "Vapor Fraction")] == 1][("Vapor", "N2O")].iloc[0]
n_dew2 = df_cvd_1[df_cvd_1[("Fractions", "Vapor Fraction")] == 1][("Total", "N2O")].iloc[0]

points = {
    "A": (initial_temp, pi, 0.5, 0.5),
    "B": (t_dew, p_dew, 0.5, 0.5),
    "C": (cold_temp, p_cold, df_idt_1[("Vapor", "N2O")].iloc[-1], df_idt_1[("Total", "N2O")].iloc[-1]),
    "D": (cold_temp, p_dew2, z_dew2, n_dew2),
    "E": (cold_temp, 0, z_dew2, n_dew2),
}


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
