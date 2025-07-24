# %%
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.path import Path
import numpy as np
from thermopack.cubic import PengRobinson, SoaveRedlichKwong
from scipy.optimize import brentq
import pandas as pd
from scipy.stats import gaussian_kde
from shapely.geometry import Point, Polygon
from matplotlib import path

from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from scipy.spatial import ConvexHull

MULTS = {"bar": 1e-5, "psi": 1 / 6894.75}

SHIFTS = {
    "C": -273.15,
}

STEPS = 1000
# %%


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


class Mixture(SoaveRedlichKwong):
    def __init__(self, comps, z: tuple, **kwargs):
        self._z = np.array(z)
        self._components = comps.split(",")
        super().__init__(comps, **kwargs)
        self._phase_envelope_cache = None  # cache envelope

    def specific_volume(self, T, P, phase=2):
        return super().specific_volume(T, P, self._z, phase=phase)[0]

    def two_phase_tpflash(self, T, P):
        return super().two_phase_tpflash(T, P, self._z)

    def two_phase_pressure(self, T, v):
        pass

    def set_z(self, z):
        self._z = np.array(z)
        self._phase_envelope_cache = None  # reset cache

    @property
    def components(self):
        return self._components

    def phase_envelope(self, p_min=1e5, t_min=0, **kwargs):
        if self._phase_envelope_cache is None:
            self._phase_envelope_cache = PhaseEnvelope(
                super().get_envelope_twophase(p_min, self._z, **kwargs),
                t_min=t_min,
                cp=self.critical(),
            )
        return self._phase_envelope_cache

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

    def isochoric_delta_T(
        self,
        p_init,
        T_init,
        T_final,
        n_temps=100,
        temp_units="C",
        pressure_units="bar",
        ax=None,
        plot_start=False,
        plot_end=False,
        **kwargs,
    ):
        """
        Change temperature from T_init to T_final at constant molar volume.
        For each temperature, solve for pressure such that molar volume is constant.
        Returns results in a pandas DataFrame indexed by pressure with multi-index columns:
        ('Total', component), ('Vapor', component), ('Liquid', component),
        ('Fractions', 'Vapor Fraction'), ('Fractions', 'Liquid Fraction'), and temperature column.
        """
        ti_k = convert_temp(T_init, temp_units, inverse=True)
        tf_k = convert_temp(T_final, temp_units, inverse=True)
        dt = (tf_k - ti_k) / n_temps
        temps = np.arange(ti_k, tf_k + dt, dt)  # Include final step

        p_init_pa = convert_pressure(p_init, pressure_units, inverse=True)

        # Initial flash at start condition to get volume reference
        flash_init = self.two_phase_tpflash(ti_k, p_init_pa)
        betaV_init = flash_init.betaV
        v_vap_init = self.specific_volume(ti_k, p_init_pa, phase=2)
        v_liq_init = self.specific_volume(ti_k, p_init_pa, phase=1)
        V_ref = betaV_init * v_vap_init + (1 - betaV_init) * v_liq_init

        pressures = []
        vapor_fractions = []
        total_compositions = []
        vapor_compositions = []
        liquid_compositions = []
        vapor_fraction_list = []
        liquid_fraction_list = []
        temperature_list = []
        regions = []

        for T in temps:

            def volume_diff(P):
                flash = self.two_phase_tpflash(T, P)
                region = self.phase_region(T, P)

                if region == 1:  # Single liquid
                    v_molar = self.specific_volume(T, P, phase=1)
                    vapor_frac = 0.0
                elif region == 2:  # Single vapor
                    v_molar = self.specific_volume(T, P, phase=2)
                    vapor_frac = 1.0
                else:  # Two-phase
                    betaV = flash.betaV
                    v_vap = self.specific_volume(T, P, phase=2)
                    v_liq = self.specific_volume(T, P, phase=1)
                    v_molar = betaV * v_vap + (1 - betaV) * v_liq
                    vapor_frac = betaV
                return v_molar - V_ref

            p_sol = brentq(volume_diff, 1e4, 1e9)
            flash = self.two_phase_tpflash(T, p_sol)
            region = self.phase_region(T, p_sol)

            if region == 1:
                vapor_frac = 0.0
                liquid_frac = 1.0
                vapor_comp = np.zeros_like(self._z)
                liquid_comp = self._z
                total_comp = self._z
            elif region == 2:
                vapor_frac = 1.0
                liquid_frac = 0.0
                vapor_comp = self._z
                liquid_comp = np.zeros_like(self._z)
                total_comp = self._z
            else:
                vapor_frac = flash.betaV
                liquid_frac = 1 - vapor_frac
                vapor_comp = flash.y
                liquid_comp = flash.x
                total_comp = vapor_frac * vapor_comp + liquid_frac * liquid_comp

            # store results in lists as before...

            pressures.append(p_sol)
            vapor_fractions.append(vapor_frac)
            total_compositions.append(total_comp)
            vapor_compositions.append(vapor_comp)
            liquid_compositions.append(liquid_comp)
            vapor_fraction_list.append(vapor_frac)
            liquid_fraction_list.append(liquid_frac)
            temperature_list.append(convert_temp(T, temp_units))
            regions.append(self.phase_region(T, p_sol))

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
            row.append(vapor_fraction_list[i])
            row.append(liquid_fraction_list[i])
            data.append(row)

        index = pd.Index(convert_pressure(np.array(pressures), pressure_units), name=f"Pressure_{pressure_units}")

        df = pd.DataFrame(data, index=index, columns=columns)

        temp_curve = f"Temperature_{temp_units}"
        # Add temperature column converted back to requested units
        df[temp_curve] = temperature_list
        df["Region"] = np.array(regions)

        if ax is not None:
            if isinstance(ax, Axes):
                ls = kwargs.pop("ls", "--")
                marker = kwargs.pop("marker", "o")
                lines = ax.plot(df[temp_curve], df.index, ls=ls, **kwargs)
                color = kwargs.pop("color", lines[0].get_color())
                if plot_start:
                    ax.plot(df[temp_curve].iloc[0], df.index[0], lw=0, marker=marker, color=color, **kwargs)
                if plot_end:
                    ax.plot(df[temp_curve].iloc[-1], df.index[-1], lw=0, marker=marker, color=color, **kwargs)

        return df

    def constant_volume_depletion(
        self,
        initial_temp,
        initial_pressure,
        initial_n_total=1.0,
        mol_fraction_to_remove=1.0,
        step_size=0.001,
        temp_units="C",
        pressure_units="bar",
        verbose=False,
        ax=None,
        plot_start=False,
        plot_end=False,
        **kwargs,
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

        fixed_moles_to_remove = step_size
        steps = int(mol_fraction_to_remove / step_size)

        # Determine initial phase region
        region = self.phase_region(T_k, P_pa)

        if verbose:
            print(f"Initial phase region: {region} (0=2Phase, 1=Liquid, 2=Vapor)")

        if region == 0:  # Two-phase region
            flash = self.two_phase_tpflash(T_k, P_pa)
            betaV = flash.betaV
            z_vapor = flash.y
            z_liquid = flash.x
        elif region == 1:  # Single liquid phase
            betaV = 0.0
            z_vapor = np.zeros_like(self._z)
            z_liquid = self._z
        elif region == 2:  # Single vapor phase
            betaV = 1.0
            z_vapor = self._z
            z_liquid = np.zeros_like(self._z)

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
        liquid_fractions = [1 - betaV]
        temperatures = [convert_temp(T_k, temp_units)]
        regions = [self.phase_region(T_k, P_pa)]
        n_total = [initial_n_total]

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
                region = self.phase_region(T_k, P)

                if region == 1:  # Single liquid
                    v_molar = self.specific_volume(T_k, P, phase=1)
                    vapor_frac = 0.0
                elif region == 2:  # Single vapor
                    v_molar = self.specific_volume(T_k, P, phase=2)
                    vapor_frac = 1.0
                else:  # Two-phase
                    flash = self.two_phase_tpflash(T_k, P)
                    betaV = flash.betaV
                    v_vap = self.specific_volume(T_k, P, phase=2)
                    v_liq = self.specific_volume(T_k, P, phase=1)
                    v_molar = betaV * v_vap + (1 - betaV) * v_liq
                    vapor_frac = betaV

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

            region = self.phase_region(T_k, P_new)

            if region == 0:  # Two-phase
                flash = self.two_phase_tpflash(T_k, P_new)
                betaV = flash.betaV
                vapor_frac = betaV
                liquid_frac = 1 - vapor_frac
                vapor_comp = flash.y
                liquid_comp = flash.x
            elif region == 1:  # Single liquid
                betaV = 0.0
                vapor_frac = 0.0
                liquid_frac = 1.0
                vapor_comp = np.zeros_like(self._z)
                liquid_comp = self._z
            else:  # Single vapor
                betaV = 1.0
                vapor_frac = 1.0
                liquid_frac = 0.0
                vapor_comp = self._z
                liquid_comp = np.zeros_like(self._z)

            if verbose:
                print(
                    f"Step {step}: Pressure = {convert_pressure(P_new, pressure_units):.2f} {pressure_units}, Vapor fraction = {betaV:.4f}"
                )


            vapor_fractions.append(vapor_frac)
            liquid_fractions.append(liquid_frac)
            total_compositions.append(vapor_frac * vapor_comp + liquid_frac * liquid_comp)
            vapor_compositions.append(vapor_comp)
            liquid_compositions.append(liquid_comp)
            temperatures.append(convert_temp(T_k, temp_units))
            pressures.append(P_new)
            regions.append(region)

            n_total.append(n_total_after)
            n_vapor = vapor_frac * n_total[-1]
            n_liquid = liquid_frac * n_total[-1]

            # Use vapor_comp and liquid_comp for updated mole counts
            n_species_vapor = n_vapor * vapor_comp
            n_species_liquid = n_liquid * liquid_comp
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
            row.append(vapor_fractions[i])
            row.append(liquid_fractions[i])
            data.append(row)

        index = pd.Index(convert_pressure(np.array(pressures), pressure_units), name=f"Pressure_{pressure_units}")

        df = pd.DataFrame(data, index=index, columns=columns)

        temp_curve = f"Temperature_{temp_units}"
        # Add temperature column converted back to requested units
        df[temp_curve]= convert_temp(np.array(temperatures), "K", inverse=True)
        df["Region"] = np.array(regions)
        df["Mol Fraction Remaining"] = np.array(n_total)

        if ax is not None:
            if isinstance(ax, Axes):
                ls = kwargs.pop("ls", "--")
                marker = kwargs.pop("marker", "o")
                lines = ax.plot(df[temp_curve], df.index, ls=ls, **kwargs)
                color = kwargs.pop("color", lines[0].get_color())
                if plot_start:
                    ax.plot(df[temp_curve].iloc[0], df.index[0], lw=0, marker=marker, color=color, **kwargs)
                if plot_end:
                    ax.plot(df[temp_curve].iloc[-1], df.index[-1], lw=0, marker=marker, color=color, **kwargs)

        return df

    def is_inside_envelope(self, T_test, P_test, debug_plot=False):
        env = self.phase_envelope(step_size=0.05)
        T_env = env._T
        P_env = env._p
        idx_ccb = np.argmax(env._p)

        # Split envelope into lower and upper branches and add (0,0) for closure
        P_lower = np.insert(P_env[: idx_ccb + 1], 0, 0)
        T_lower = np.insert(T_env[: idx_ccb + 1], 0, 0)

        P_upper = np.append(P_env[idx_ccb:], 0)  # append zero
        T_upper = np.append(T_env[idx_ccb:], 0)  # append zero

        P_ccb = P_env[idx_ccb]

        if P_test > P_ccb:
            inside = False
        else:
            # Interpolate temperatures on both branches at P_test
            T_low = np.interp(P_test, P_lower, T_lower)
            T_high = np.interp(P_test, P_upper, T_upper)

            # Check if T_test lies between envelope temperatures (account for order)
            inside = (T_low <= T_test <= T_high) or (T_high <= T_test <= T_low)

        if debug_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(T_lower, P_lower, label="Lower Envelope", color="blue")
            plt.plot(T_upper, P_upper, label="Upper Envelope", color="darkred")
            color = "green" if inside else "red"
            plt.plot(
                T_test,
                P_test,
                "*",
                markersize=15,
                label=f"Test Point ({'Inside' if inside else 'Outside'})",
                color=color,
            )
            plt.xlabel("Temperature [K]")
            plt.ylabel("Pressure [Pa]")
            plt.title("Phase Envelope and Test Point")
            plt.legend()
            plt.grid(True)
            plt.show()

        return inside

    def phase_region(self, T, P):
        """
        Return:
            0 for two-phase (inside envelope)
            1 for vapor phase (outside envelope, vapor-like)
            2 for liquid phase (outside envelope, liquid-like)
        """
        env = self.phase_envelope()
        if self.is_inside_envelope(T, P):
            return 0  # Two-phase

        Tc = env.Tc(units="K")
        Pc = env.pc(units="Pa")

        if T >= Tc and P <= Pc:
            return 2  # Vapor
        elif T <= Tc and P >= Pc:
            return 1  # Liquid
        else:
            return 2  # Default to vapor for supercritical/complex


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

# %%
entonox = Mixture("N2O,O2", (0.5, 0.5))
env = entonox.phase_envelope(step_size=0.1, t_min=273.15 - 40)

cold_temp = -25
pi = 3000 / 14.50377  # bar gauge pressure
cold_usage = 0.40
warm_temp = 0


fig = entonox.plot_gas_fraction("N2O", t_min=cold_temp-5, cmap="rainbow_r", vmin=0.3)
fig = env.plot(fig=fig, xlim=(cold_temp-5, 25))
ax = fig.axes[0]


pi_pa = convert_pressure(pi, "bar", inverse=True)
df_idt_1 = entonox.isochoric_delta_T(pi, 20, cold_temp, ax=ax, plot_start=True, plot_end=False)
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6), layout="tight")
df_idt_1.plot(y=("Vapor", "N2O"), label=f"Cooling to {cold_temp:0.0f} C", ls="--", ax=ax2)
ax2.set_xlim(200, 0)

# Initial flash conditions at cold temperature
p_cold = df_idt_1.index[-1]

# Initally assume that all the gas is used at -20C
df_cvd_1 = entonox.constant_volume_depletion(
    initial_temp=cold_temp,
    initial_pressure=p_cold,
    initial_n_total=1.0,
    mol_fraction_to_remove=1.0,
    ax=ax,
    plot_start=True,
    plot_end=False,
)
df_cvd_1.plot(
    y=("Vapor", "N2O"),
    ax=ax2,
    label=f"Dispensing at {cold_temp:0.0f} C",
)


# Now assume that 50% of the gas is used at -20C and that the cyclinder is then warmed
# to 0C and the rest us used


idx = df_cvd_1[df_cvd_1["Mol Fraction Remaining"] <= 1 - cold_usage].index[0]
z = df_cvd_1.loc[idx]["Total"]
entonox.set_z(z.to_list())
p_init = idx
T_init = float(df_cvd_1.loc[idx]["Temperature_C"].iloc[0])

df_idt_2 = entonox.isochoric_delta_T(p_init, T_init, warm_temp, ax=ax, ls='-')
df_idt_2.plot(y=("Vapor", "N2O"), ax=ax2, label=f"Warming {cold_temp:0.0f} to {warm_temp:0.0f} C", ls="--")
entonox.phase_envelope(step_size=0.01).plot(
    fig=fig, split=False, lw=1, label=f"Phase Envelop After 50% @ {cold_temp:0.0f}C", color="purple", cp_color="purple"
)
p_warm = df_idt_2.index[-1]

df_cvd_2 = entonox.constant_volume_depletion(
    initial_temp=warm_temp,
    initial_pressure=p_warm,
    initial_n_total=1 - cold_usage,
    mol_fraction_to_remove=1 - cold_usage,
    ax=ax,
    plot_start=True,
    plot_end=False,
)
df_cvd_2.plot(y=("Vapor", "N2O"), ax=ax2, label=f"Dispensing at {warm_temp:0.0f} C")
ax2.set_xlabel("Gauge Pressure [bara]")
ax2.set_ylabel("Vapour Phase N2O Concentration [dimensionless]")
ax2.set_ylim(0, 1)
ax2.set_title(f"Gas Concentration vs Gauge Pressure - Initial Pressure = {pi:0.0f} bara")
# %%
