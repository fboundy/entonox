# %%
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from thermopack.cubic import PengRobinson
from scipy.optimize import brentq
from copy import deepcopy

MULTS = {"bar": 1e-5, "psi": 1 / 6894.75}

SHIFTS = {
    "C": -273.15,
}

STEPS = 1000


def convert_temp(temp, units, inverse=False):
    shift = SHIFTS.get(units, 0)
    if inverse:
        shift = -shift
    return temp + shift


def convert_pressure(pressure, units, inverse=False):
    mult = MULTS.get(units, 1)
    if inverse:
        mult = 1 / mult
    return pressure * mult


class Mixture(PengRobinson):
    def __init__(self, comps, z: tuple, **kwargs):
        self._z = np.array(z)
        self._components = comps.split(",")
        super().__init__(comps, **kwargs)

    def specific_volume(self, T, P, phase=2):
        return super().specific_volume(T, P, self._z, phase=phase)[0]

    def two_phase_tpflash(self, T, P):
        return super().two_phase_tpflash(T, P, self._z)

    def two_phase_pressure(self, T, v):
        pass

    # Add this method:
    def set_z(self, z):
        self._z = np.array(z)

    @property
    def components(self):
        return self._components

    def phase_envelope(self, p_min=1e5, t_min=0, **kwargs):
        return PhaseEnvelope(
            super().get_envelope_twophase(p_min, self._z, **kwargs),
            t_min=t_min,
            cp=self.critical(),
        )

    def critical(self, **kwargs):
        return super().critical(self._z, **kwargs)

    def plot_gas_fraction(
        self,
        component,
        fig=None,
        figsize=(8, 6),
        layout="tight",
        temp_units="C",
        pressure_units="bar",
        n_temp=200,
        n_press=200,
        t_min=None,
        **kwargs,
    ):

        if component not in self.components:
            raise ValueError(f"No such component: {component}")
        else:
            idx = self.components.index(component)

        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, layout=layout)
        else:
            ax = fig.axes[0]

        if t_min is not None:
            t_min = convert_temp(t_min, temp_units, inverse=True)
        else:
            t_min = self.phase_envelope().t_min

        p_min = self.phase_envelope().p_min
        p_max = self.phase_envelope().p_max
        t_max = self.phase_envelope().t_max

        temp = np.arange(t_min, t_max, (t_max - t_min) / n_temp)
        press = np.arange(p_min, p_max, (p_max - p_min) / n_press)
        # V = np.full((len(press), len(temp)), np.nan)
        N = np.full((len(press), len(temp)), np.nan)
        for i, T in enumerate(temp):
            for j, p in enumerate(press):
                f = self.two_phase_tpflash(T, p)
                # if f.betaV >= 0:
                # V[j, i] = f.betaV
                if f.y[idx] > 0:
                    N[j, i] = f.y[idx]
        T, P = np.meshgrid(temp, press)
        # ax.contour(T, P, V, levels=10)
        # # print(T)
        # countours = ax.contour(convert_temp(T,temp_units), convert_pressure(P,pressure_units), N, levels=10)
        # img = ax.imshow(N)
        img = ax.imshow(
            N,
            extent=(
                convert_temp(t_min, temp_units),
                convert_temp(t_max, temp_units),
                convert_pressure(p_min, pressure_units),
                convert_pressure(p_max, pressure_units),
            ),
            aspect="auto",
            origin="lower",
            **kwargs,
        )
        fig.colorbar(img, label=f"% {component} in Gas Phase")

        return fig

    def cool(self, p_init, T_init, T_final, n_temps=100, temp_units="C", pressure_units="bar"):
        press = []
        ti_k = convert_temp(T_init, units=temp_units, inverse=True)
        tf_k = convert_temp(T_final, units=temp_units, inverse=True)
        dt = (tf_k - ti_k) / n_temps

        temp = np.arange(ti_k, tf_k, dt)

        y = []
        v = []

        pi_pa = convert_pressure(p_init, pressure_units, inverse=True)

        flash_init = self.two_phase_tpflash(ti_k, pi_pa)
        betaV_init = flash_init.betaV
        v_vap_init = self.specific_volume(ti_k, pi_pa, phase=2)
        v_liq_init = self.specific_volume(ti_k, pi_pa, phase=1)
        V_ref = betaV_init * v_vap_init + (1 - betaV_init) * v_liq_init

        for T in temp:

            def volume_diff(P):
                flash = self.two_phase_tpflash(T, P)
                betaV = flash.betaV
                v_vap = self.specific_volume(T, P, phase=2)
                v_liq = self.specific_volume(T, P, phase=1)
                v_molar = betaV * v_vap + (1 - betaV) * v_liq
                return v_molar - V_ref

            try:
                p_sol = brentq(volume_diff, 1e4, 1e9)
                press.append(p_sol)
            except ValueError:
                press.append(np.nan)

            if np.isnan(press[-1]):
                y.append(np.nan)
                v.append(np.nan)
            else:
                f = self.two_phase_tpflash(T, p_sol)
                y.append(f.y)
                v.append(f.betaV)

        return (
            convert_temp(temp, temp_units),
            convert_pressure(np.array(press), pressure_units),
            np.array(y),
            np.array(v),
        )


class PhaseEnvelope:
    def __init__(self, Tp, cp, t_min=0):
        self._T = Tp[0][Tp[0] > t_min]
        self._p = Tp[1][Tp[0] > t_min]
        self._Tc = cp[0]
        self._pc = cp[2]
        self._t_min = t_min

    def T(self, units="K"):
        return convert_temp(self._T, units)

    def Tc(self, units="K"):
        return convert_temp(self._Tc, units)

    def p(self, units="Pa"):
        return convert_pressure(self._p, units)

    def pc(self, units="Pa"):
        return convert_pressure(self._pc, units)

    @property
    def t_min(self):
        return self._t_min

    @property
    def t_max(self):
        return np.nanmax(self._T)

    @property
    def p_min(self):
        return np.nanmin(self._p)

    @property
    def p_max(self):
        return np.nanmax(self._p)

    def plot(
        self,
        fig=None,
        figsize=(8, 6),
        layout="tight",
        temp_units="C",
        pressure_units="bar",
        cp=True,
        split=True,
        xlim=None,
        ylim=None,
        cp_color=None,
        **kwargs,
    ):
        kwargs["lw"] = kwargs.get("lw", 3)

        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, layout=layout)
        else:
            ax = fig.axes[0]

        if split:
            bp = self._bubble_point(temp_units, pressure_units)
            ax.plot(bp[0], bp[1], label="Bubble Point", color="darkgreen", **kwargs)
            dp = self._dew_point(temp_units, pressure_units)
            ax.plot(dp[0], dp[1], label="Dew Point", color="darkred", **kwargs)
            # ax.plot(self._dew_point(temp_units, pressure_units), **kwargs)

        else:
            ax.plot(self.T(temp_units), self.p(pressure_units), **kwargs)

        if cp:
            if cp_color is None:
                cp_color = "orange"
            ax.plot(
                self.Tc(temp_units),
                self.pc(pressure_units),
                marker="*",
                label=f"Critical Point ({self.Tc(units=temp_units):0.1f} {temp_units}, {self.pc(units=pressure_units):0.1f} {pressure_units})",
                color=cp_color,
                markersize=20,
                lw=0,
            )

        ax.set_xlabel(f"Temperature [{temp_units}]")
        ax.set_ylabel(f"Pressure [{pressure_units}]")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        return fig

    def _bubble_point(self, temp_units="K", pressure_units="Pa"):
        idx_crit = self._cp_index
        T_bub = self._T[idx_crit:]
        p_bub = self._p[idx_crit:]
        return convert_temp(T_bub, temp_units), convert_pressure(p_bub, pressure_units)

    def _dew_point(self, temp_units="K", pressure_units="Pa"):
        idx_crit = self._cp_index
        T_dew = self._T[: idx_crit + 1]
        p_dew = self._p[: idx_crit + 1]
        return convert_temp(T_dew, temp_units), convert_pressure(p_dew, pressure_units)

    @property
    def _cp_index(self):
        # Normalize differences for T and P to be on comparable scales
        T_diff_norm = np.abs(self._T - self._Tc) / self._Tc
        P_diff_norm = np.abs(self._p - self._pc) / self._pc
        combined_diff = np.sqrt(T_diff_norm**2 + P_diff_norm**2)

        return np.argmin(combined_diff)


entonox = Mixture("N2O,O2", (0.5, 0.5))
env = entonox.phase_envelope(step_size=0.1, t_min=273.15 - 40)

fig = entonox.plot_gas_fraction("N2O", t_min=-40, cmap="rainbow_r")
fig = env.plot(fig=fig, xlim=(-30, 25))
ax = fig.axes[0]

cold_temp = -20
pi = 200
warm_from = 55


i = 0

color = f"C{i}"
fig1, ax1 = plt.subplots()

removed_vapor_composition = []
liquid_n2o_conc = []

entonox = Mixture("N2O,O2", (0.5, 0.5))
f = entonox.two_phase_tpflash(convert_temp(20, "C", True), pi)
# %%
t, p, y, v1 = entonox.cool(pi, 20, cold_temp)
ax.plot(t, p, ls="--", label=f"Cooling Curve: Initial Pressure = {pi} bar")

# Starting state from initial flash
idx_cold = np.argmin(np.abs(t - cold_temp))
p_cold = convert_pressure(p[idx_cold], "bar", inverse=True)

f = entonox.two_phase_tpflash(convert_temp(cold_temp, "C", True), p_cold)
# %%
n_total = 1.0  # initial mole basis

betaV = f.betaV
z_vapor = f.y
z_liquid = f.x

n_vapor = betaV * n_total
n_liquid = (1 - betaV) * n_total

n_species_vapor = n_vapor * z_vapor
n_species_liquid = n_liquid * z_liquid

v_vap = entonox.specific_volume(convert_temp(cold_temp, "C", True), p_cold, phase=2)
v_liq = entonox.specific_volume(convert_temp(cold_temp, "C", True), p_cold, phase=1)

V_ref = betaV * v_vap + (1 - betaV) * v_liq
V_total = V_ref * n_total

pressures = [p_cold]
vapor_fractions = [betaV]
overall_compositions = [betaV * z_vapor + (1 - betaV) * z_liquid]
removed_vapor_composition = []
liquid_n2o_conc = []

fixed_moles_to_remove = n_total / STEPS  # Remove 1% vapor per step
max_steps = STEPS
z = []
n = []

for step in range(max_steps):
    # print(f"Step {step} Flash result before removal:")
    # print(f"Pressure: {convert_pressure(pressures[-1], 'bar'):.2f} bar, Vapor fraction: {vapor_fractions[-1]:.4f}")

    total_vapor_moles = np.sum(n_species_vapor)
    if total_vapor_moles <= 0:
        # print(f"No vapor left to remove at step {step}. Stopping.")
        break

    moles_to_remove = min(fixed_moles_to_remove, total_vapor_moles)

    # Remove vapor moles proportionally by composition
    removed_vapor_moles = n_species_vapor * (moles_to_remove / total_vapor_moles)
    removed_total_moles = np.sum(removed_vapor_moles)
    removed_vapor_composition.append(removed_vapor_moles / removed_total_moles)

    # Subtract removed vapor moles
    n_species_vapor -= removed_vapor_moles

    # Update total moles after removal
    n_total_after = np.sum(n_species_vapor) + np.sum(n_species_liquid)
    n_species_after = n_species_vapor + n_species_liquid

    def volume_diff(P):
        flash = entonox.two_phase_tpflash(convert_temp(cold_temp, "C", True), P)
        betaV = flash.betaV

        if betaV < 0 or betaV < 1e-6:
            # Single liquid phase or no vapor phase
            v_molar = entonox.specific_volume(convert_temp(cold_temp, "C", True), P, phase=1)
        elif betaV > 1 - 1e-6:
            # Single vapor phase
            v_molar = entonox.specific_volume(convert_temp(cold_temp, "C", True), P, phase=2)
        else:
            # Two-phase mixture
            v_vap = entonox.specific_volume(convert_temp(cold_temp, "C", True), P, phase=2)
            v_liq = entonox.specific_volume(convert_temp(cold_temp, "C", True), P, phase=1)
            v_molar = betaV * v_vap + (1 - betaV) * v_liq

        return v_molar * n_total_after - V_total

    # Find new pressure that keeps total volume constant
    a = pressures[-1] * 0.8
    b = pressures[-1] * 1.2

    fa = volume_diff(a)
    fb = volume_diff(b)

    if np.isnan(fa) or np.isnan(fb) or fa * fb > 0:
        # print(f"Warning: No root bracket found at step {step}, trying wider interval...")
        a = pressures[-1] * 0.1
        b = pressures[-1] * 10
        fa = volume_diff(a)
        fb = volume_diff(b)
        if fa * fb > 0:
            # print("Still no bracket in wider interval. Stopping iteration.")
            break

    P_new = brentq(volume_diff, a, b)

    # Update mole fractions
    z_new = n_species_after / np.sum(n_species_after)

    entonox.set_z(z_new)

    # Recalculate flash at new pressure
    f = entonox.two_phase_tpflash(convert_temp(cold_temp, "C", True), P_new)

    betaV = f.betaV
    # print(f"Step {step} Flash result after removal:")
    # print(f"Pressure: {convert_pressure(P_new, 'bar'):.2f} bar, Vapor fraction: {betaV:.4f}")

    if betaV < 1e-6 or betaV > 0.99:
        # print("Vapor fraction near zero or one, stopping iteration.")
        break

    z_vapor = f.y
    z_liquid = f.x
    liquid_n2o_conc.append(z_liquid[0])  # Track liquid N2O mole fraction

    n_total = n_total_after
    n_vapor = betaV * n_total
    n_liquid = (1 - betaV) * n_total

    z.append(z_new)
    n.append(n_total)

    n_species_vapor = n_vapor * z_vapor
    n_species_liquid = n_liquid * z_liquid

    pressures.append(P_new)
    vapor_fractions.append(betaV)
    overall_compositions.append(betaV * z_vapor + (1 - betaV) * z_liquid)


removed_array = np.array(removed_vapor_composition)

ax1.plot(
    convert_pressure(np.array(pressures), units="bar"),
    removed_array[:, 0],
    label=f"Gas (Initial Pressure = {pi} bar)",
    ls="--",
    color="black",
)
ax1.plot(
    convert_pressure(np.array(pressures[1:]), units="bar"),
    liquid_n2o_conc,
    label=f"Liquid (Initial Pressure = {pi} bar)",
    ls="-",
    color="black",
)

ax1.plot(
    convert_pressure(np.array(pressures), units="bar"),
    np.array(overall_compositions)[:, 0],
    label=f"Overall (Initial Pressure = {pi} bar)",
    ls="-",
    color=color,
)

ax1.set_xlabel("Gauge Pressure [bar]")
ax1.set_ylabel("N2O concentration")
# ax1[1].set_ylabel("Residual liquid N2O concentration")
ax1.set_xlim(200, 0)
ax1.set_title(f"Composition after cooling to {cold_temp} C")
ax.legend()
ax1.legend(ncols=1, fontsize=8)

frac_l = 50
frac_h = 60
frac_step = 10


for i, w in enumerate(range(int(frac_l * STEPS / 100), int(frac_h * STEPS / 100), int(frac_step * STEPS / 100))):
    color = f"C{i+1}"
    entonox.set_z(z[w])
    n_total = n[w]
    env = entonox.phase_envelope(step_size=0.1, t_min=273.15 - 40)
    env.plot(fig=fig, xlim=(-30, 25), ylim=(0, 200), split=False, color=color, cp_color=color, lw=1)

    pi = convert_pressure(pressures[w], "bar")

    warm_temp = 0

    t, p, y, v1 = entonox.cool(pi, cold_temp, warm_temp, temp_units="C")
    ax.plot(t, p, ls="--", label=f"w={w}", color=color)
    # ax2.plot(t, p, ls="--", label=f"Warming Curve: Initial Pressure = {pi} bar")
    f = entonox.two_phase_tpflash(convert_temp(t[0], "C", True), convert_pressure(p[0], "bar", True))
    print(f"Flash results for w={w}")
    print(f)

    idx_warm = np.argmin(np.abs(t - warm_temp))
    p_warm = convert_pressure(p[idx_warm], "bar", inverse=True)
    f = entonox.two_phase_tpflash(convert_temp(warm_temp, "C", True), p_warm)

    betaV = f.betaV
    z_vapor = f.y
    z_liquid = f.x

    n_vapor = betaV * n_total
    n_liquid = (1 - betaV) * n_total

    n_species_vapor = n_vapor * z_vapor
    n_species_liquid = n_liquid * z_liquid

    v_vap = entonox.specific_volume(convert_temp(warm_temp, "C", True), p_warm, phase=2)
    v_liq = entonox.specific_volume(convert_temp(warm_temp, "C", True), p_warm, phase=1)

    V_ref = betaV * v_vap + (1 - betaV) * v_liq
    V_total = V_ref * n_total

    pressures2 = [pressures[w], p_warm]
    vapor_fractions = [betaV]
    overall_compositions2 = overall_compositions[w : w + 1] + [betaV * z_vapor + (1 - betaV) * z_liquid]
    removed_vapor_composition2 = removed_vapor_composition[w : w + 1]
    liquid_n2o_conc2 = liquid_n2o_conc[w : w + 1]

    print(f)
    max_steps = STEPS

    for step in range(w, max_steps):
        total_vapor_moles = np.sum(n_species_vapor)
        if total_vapor_moles <= 0:
            print(f"No vapor left to remove at step {step}. Stopping.")
            break

        moles_to_remove = min(fixed_moles_to_remove, total_vapor_moles)

        # Remove vapor moles proportionally by composition
        removed_vapor_moles = n_species_vapor * (moles_to_remove / total_vapor_moles)
        removed_total_moles = np.sum(removed_vapor_moles)
        removed_vapor_composition2.append(removed_vapor_moles / removed_total_moles)

        # Subtract removed vapor moles
        n_species_vapor -= removed_vapor_moles

        # Update total moles after removal
        n_total_after = np.sum(n_species_vapor) + np.sum(n_species_liquid)
        n_species_after = n_species_vapor + n_species_liquid

        def volume_diff(P):
            flash = entonox.two_phase_tpflash(convert_temp(warm_temp, "C", True), P)
            betaV = flash.betaV

            if betaV < 1e-6:
                # Single liquid phase or no vapor phase
                v_molar = entonox.specific_volume(convert_temp(warm_temp, "C", True), P, phase=1)
            elif betaV > 1 - 1e-6 or betaV == -1:
                # Single vapor phase
                v_molar = entonox.specific_volume(convert_temp(warm_temp, "C", True), P, phase=2)
            else:
                # Two-phase mixture
                v_vap = entonox.specific_volume(convert_temp(warm_temp, "C", True), P, phase=2)
                v_liq = entonox.specific_volume(convert_temp(warm_temp, "C", True), P, phase=1)
                v_molar = betaV * v_vap + (1 - betaV) * v_liq

            return v_molar * n_total_after - V_total

        # Find new pressure that keeps total volume constant
        a = pressures2[-1] * 0.8
        b = pressures2[-1] * 0.99

        fa = volume_diff(a)
        fb = volume_diff(b)

        if np.isnan(fa) or np.isnan(fb) or fa * fb > 0:
            print(f"Warning: No root bracket found at step {step}, trying wider interval...")
            a = pressures2[-1] * 0.1
            b = pressures2[-1] * 10
            fa = volume_diff(a)
            fb = volume_diff(b)
            if fa * fb > 0:
                print("Still no bracket in wider interval. Stopping iteration.")
                break

        P_new = brentq(volume_diff, a, b)

        # Update mole fractions
        z_new = n_species_after / np.sum(n_species_after)

        entonox.set_z(z_new)

        # Recalculate flash at new pressure
        f = entonox.two_phase_tpflash(convert_temp(warm_temp, "C", True), P_new)
        print(step, convert_pressure(P_new, "bar"), betaV)

        betaV = f.betaV
        # print(f"Step {step} Flash result after removal:")
        # print(f"Pressure: {convert_pressure(P_new, 'bar'):.2f} bar, Vapor fraction: {betaV:.4f}")

        if betaV < 1e-6 and betaV > 0:
            print(f"Vapor fraction near zero ({betaV:0.4f}), stopping iteration.")
            break

        z_vapor = f.y
        z_liquid = f.x
        liquid_n2o_conc2.append(z_liquid[0])  # Track liquid N2O mole fraction

        n_total = n_total_after
        n_vapor = betaV * n_total
        n_liquid = (1 - betaV) * n_total

        n_species_vapor = n_vapor * z_vapor
        n_species_liquid = n_liquid * z_liquid

        pressures2.append(P_new)
        vapor_fractions.append(betaV)
        overall_compositions2.append(betaV * z_vapor + (1 - betaV) * z_liquid)

    removed_array = np.array(removed_vapor_composition2)

    ax1.plot(
        convert_pressure(np.array(pressures2[:-1]), units="bar"),
        removed_array[:, 0],
        label=f"Gas (Initial Pressure = {pi} bar)",
        ls="--",
        color="red",
    )
    ax1.plot(
        convert_pressure(np.array(pressures2[:-1]), units="bar"),
        liquid_n2o_conc2,
        label=f"Liquid (Initial Pressure = {pi} bar)",
        ls="-",
        color="red",
    )

    ax1.plot(
        convert_pressure(np.array(pressures2), units="bar"),
        np.array(overall_compositions2)[:, 0],
        label=f"Overall (Initial Pressure = {pi} bar)",
        ls="-",
        color=color,
    )

    ax1.set_xlabel("Gauge Pressure [bar]")
    ax1.set_ylabel("N2O concentration")
    # ax1[1].set_ylabel("Residual liquid N2O concentration")
    ax1.set_xlim(200, 0)
    ax1.set_title(f"Composition after warming to {warm_temp} C")

    env = entonox.phase_envelope(step_size=0.1, t_min=273.15 - 40)
    env.plot(fig=fig, xlim=(-30, 25), ylim=(0, 200), split=False, color="red", cp_color="red", lw=1)
    ax.plot([warm_temp] * (len(pressures2) - 1), convert_pressure(np.array(pressures2[1:]), "bar"), ls="-.")
ax.legend(ncols=2, fontsize=8, loc="upper center")
# %%
