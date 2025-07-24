# %%
import matplotlib.pyplot as plt
import numpy as np
from thermopack.cubic import PengRobinson
from scipy.optimize import brentq
import pandas as pd

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
        N = np.full((len(press), len(temp)), np.nan)
        for i, T in enumerate(temp):
            for j, p in enumerate(press):
                f = self.two_phase_tpflash(T, p)
                if f.y[idx] > 0:
                    N[j, i] = f.y[idx]
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

    def isochoric_delta_T(self, p_init, T_init, T_final, n_temps=100, temp_units="C", pressure_units="bar"):
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

    def constant_volume_depletion(
        self,
        initial_temp,
        initial_pressure,
        initial_n_total=1.0,
        mol_fraction_to_remove=1.0,
        steps=1000,
        temp_units="C",
        pressure_units="bar",
        verbose=False,
    ):
        """
        Remove vapor moles stepwise at constant volume and temperature.
        Inputs:
        - initial_temp: in temp_units (default 'C')
        - initial_pressure: in pressure_units (default 'bar')
        - initial_n_total: initial total moles
        - mol_fraction_to_remove: fraction of moles to remove over steps
        - steps: number of steps
        - temp_units: units for temperature input/output (default 'C')
        - pressure_units: units for pressure input/output (default 'bar')
        Returns:
        - pandas DataFrame indexed by pressure (converted to pressure_units)
            with MultiIndex columns:
            ('Total', component), ('Vapor', component), ('Liquid', component),
            ('Fractions', 'Vapor Fraction'), ('Fractions', 'Liquid Fraction')
        - Temperature column in temp_units
        """
        # Convert inputs to internal units (K, Pa)
        T_k = convert_temp(initial_temp, temp_units, inverse=True)
        P_pa = convert_pressure(initial_pressure, pressure_units, inverse=True)

        fixed_moles_to_remove = mol_fraction_to_remove / steps

        flash = self.two_phase_tpflash(T_k, P_pa)
        betaV = flash.betaV
        z_vapor = flash.y
        z_liquid = flash.x

        n_vapor = betaV * initial_n_total
        n_liquid = (1 - betaV) * initial_n_total

        n_species_vapor = n_vapor * z_vapor
        n_species_liquid = n_liquid * z_liquid

        v_vap = self.specific_volume(T_k, P_pa, phase=2)
        v_liq = self.specific_volume(T_k, P_pa, phase=1)
        V_total = (betaV * v_vap + (1 - betaV) * v_liq) * initial_n_total

        pressures = [P_pa]
        vapor_fractions = [betaV]
        total_compositions = [betaV * z_vapor + (1 - betaV) * z_liquid]
        vapor_compositions = [z_vapor]
        liquid_compositions = [z_liquid]
        vapor_fractions_list = [betaV]
        liquid_fractions_list = [1 - betaV]
        temperatures = [T_k]

        n_total = initial_n_total

        for step in range(steps):
            total_vapor_moles = np.sum(n_species_vapor)
            if total_vapor_moles <= 0:
                if verbose:
                    print(f"No vapor left to remove at step {step}. Stopping.")
                break

            moles_to_remove = min(fixed_moles_to_remove, total_vapor_moles)
            removed_vapor_moles = n_species_vapor * (moles_to_remove / total_vapor_moles)
            n_species_vapor -= removed_vapor_moles

            n_total_after = np.sum(n_species_vapor) + np.sum(n_species_liquid)
            n_species_after = n_species_vapor + n_species_liquid

            def volume_diff(P):
                flash = self.two_phase_tpflash(T_k, P)
                betaV = flash.betaV

                if betaV < 1e-6:
                    v_molar = self.specific_volume(T_k, P, phase=1)
                elif betaV > 1 - 1e-6 or betaV == -1:
                    v_molar = self.specific_volume(T_k, P, phase=2)
                else:
                    v_vap = self.specific_volume(T_k, P, phase=2)
                    v_liq = self.specific_volume(T_k, P, phase=1)
                    v_molar = betaV * v_vap + (1 - betaV) * v_liq

                return v_molar * n_total_after - V_total

            a = pressures[-1] * 0.8
            b = pressures[-1] * 1.2

            fa = volume_diff(a)
            fb = volume_diff(b)

            if np.isnan(fa) or np.isnan(fb) or fa * fb > 0:
                if verbose:
                    print(f"Warning: No root bracket found at step {step}, trying wider interval...")
                a = pressures[-1] * 0.1
                b = pressures[-1] * 10
                fa = volume_diff(a)
                fb = volume_diff(b)
                if fa * fb > 0:
                    if verbose:
                        print("Still no bracket in wider interval. Stopping iteration.")
                    break

            P_new = brentq(volume_diff, a, b)

            z_new = n_species_after / np.sum(n_species_after)
            self.set_z(z_new)

            flash = self.two_phase_tpflash(T_k, P_new)
            betaV = flash.betaV
            if verbose:
                print(
                    f"Step {step}: Pressure = {convert_pressure(P_new, pressure_units):.2f} {pressure_units}, Vapor fraction = {betaV:.4f}"
                )

            if betaV < 1e-6 or betaV > 0.99:
                if verbose:
                    print("Vapor fraction near zero or one, stopping iteration.")
                break

            z_vapor = flash.y
            z_liquid = flash.x

            vapor_fractions.append(betaV)
            total_compositions.append(betaV * z_vapor + (1 - betaV) * z_liquid)
            vapor_compositions.append(z_vapor)
            liquid_compositions.append(z_liquid)
            vapor_fractions_list.append(betaV)
            liquid_fractions_list.append(1 - betaV)
            temperatures.append(T_k)
            pressures.append(P_new)

            n_total = n_total_after
            n_vapor = betaV * n_total
            n_liquid = (1 - betaV) * n_total

            n_species_vapor = n_vapor * z_vapor
            n_species_liquid = n_liquid * z_liquid

        # Prepare DataFrame columns with multiindex
        columns = pd.MultiIndex.from_tuples(
            [(phase, comp) for phase in ["Total", "Vapor", "Liquid"] for comp in self.components]
            + [("Fractions", "Vapor Fraction"), ("Fractions", "Liquid Fraction")],
            names=["Phase", "Component"],
        )

        data = []
        for i in range(len(pressures)):
            row = []
            row.extend(total_compositions[i])
            row.extend(vapor_compositions[i])
            row.extend(liquid_compositions[i])
            row.append(vapor_fractions_list[i])
            row.append(liquid_fractions_list[i])
            data.append(row)

        index = pd.Index(convert_pressure(np.array(pressures), pressure_units), name=f"Pressure_{pressure_units}")

        df = pd.DataFrame(data, index=index, columns=columns)

        # Add temperature column converted back to requested units
        df["Temperature_" + temp_units] = convert_temp(np.array(temperatures), "K", inverse=True)

        return df


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
        T_diff_norm = np.abs(self._T - self._Tc) / self._Tc
        P_diff_norm = np.abs(self._p - self._pc) / self._pc
        combined_diff = np.sqrt(T_diff_norm**2 + P_diff_norm**2)

        return np.argmin(combined_diff)


# ------------------ USAGE ------------------

entonox = Mixture("N2O,O2", (0.5, 0.5))
env = entonox.phase_envelope(step_size=0.1, t_min=273.15 - 40)

fig = entonox.plot_gas_fraction("N2O", t_min=-40, cmap="rainbow_r", vmin=0.3)
fig = env.plot(fig=fig, xlim=(-25, 25))
ax = fig.axes[0]

cold_temp = -20
pi = 200  # bar gauge pressure
pi_pa = convert_pressure(pi, "bar", inverse=True)

t, p, y, v1 = entonox.isochoric_delta_T(pi, 20, cold_temp)
ax.plot(t, p, ls="--", label=f"Cooling Curve: Initial Pressure = {pi} bar")

# Initial flash conditions at cold temperature
idx_cold = np.argmin(np.abs(t - cold_temp))
p_cold = p[idx_cold]

df_cvd_1 = entonox.constant_volume_depletion(
    initial_temp=cold_temp,
    initial_pressure=p_cold,
    initial_n_total=1.0,
    mol_fraction_to_remove=1.0,
    steps=1000,
)

# %%
ax1 = plt.figure().gca()
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
    color="C0",
)
ax1.set_xlabel("Gauge Pressure [bar]")
ax1.set_ylabel("N2O concentration")
ax1.set_xlim(200, 0)
ax1.set_title(f"Composition after cooling to {cold_temp} C")
ax1.legend(ncols=1, fontsize=8)
ax.legend()

# Add any additional analysis or plotting here

# %%
