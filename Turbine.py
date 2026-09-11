"""
Turbine performance module.

Given inlet conditions (fluid, T, P) and outlet pressure, compute power output
using user-specified isentropic and mechanical efficiencies.

Edit the INPUTS block below and run the file.

Requires:  pip install CoolProp matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI, PhaseSI


# ======================================================================
#                           USER INPUTS
# ======================================================================
fluid          = 'Isobutane'    # CoolProp fluid: Water, R134a, R245fa, CO2, Air, ...
T_in_C         = 40     # Inlet temperature [degC]
P_in_bar       = 4      # Inlet pressure    [bar]
P_out_bar      = 1       # Outlet pressure   [bar]
m_dot          = 50.0       # Mass flow rate    [kg/s]
eta_isentropic = 0.88       # Isentropic efficiency (0-1)
eta_mechanical = 0.98       # Mechanical efficiency (0-1)
# ======================================================================


def turbine(fluid, T_in_C, P_in_bar, P_out_bar, m_dot,
            eta_isentropic, eta_mechanical):
    """Compute turbine states and powers."""
    T_in = T_in_C + 273.15
    P_in = P_in_bar * 1e5
    P_out = P_out_bar * 1e5

    if P_out >= P_in:
        raise ValueError("Outlet pressure must be lower than inlet pressure.")
    if not (0 < eta_isentropic <= 1) or not (0 < eta_mechanical <= 1):
        raise ValueError("Efficiencies must be in (0, 1].")

    # State 1 - inlet
    h1 = PropsSI('H', 'T', T_in, 'P', P_in, fluid)
    s1 = PropsSI('S', 'T', T_in, 'P', P_in, fluid)
    rho1 = PropsSI('D', 'T', T_in, 'P', P_in, fluid)
    phase_in = PhaseSI('T', T_in, 'P', P_in, fluid)

    # State 2s - ideal isentropic outlet
    h2s = PropsSI('H', 'P', P_out, 'S', s1, fluid)
    T2s = PropsSI('T', 'P', P_out, 'S', s1, fluid)

    # State 2 - actual outlet (turbine isentropic efficiency definition)
    dh_isen = h1 - h2s
    dh_act  = eta_isentropic * dh_isen
    h2 = h1 - dh_act
    T2 = PropsSI('T', 'P', P_out, 'H', h2, fluid)
    s2 = PropsSI('S', 'P', P_out, 'H', h2, fluid)
    phase_out = PhaseSI('P', P_out, 'H', h2, fluid)
    try:
        x2 = PropsSI('Q', 'P', P_out, 'H', h2, fluid)
    except Exception:
        x2 = None

    # Powers
    W_isen  = m_dot * dh_isen
    W_shaft = m_dot * dh_act
    W_out   = W_shaft * eta_mechanical
    eta_overall = eta_isentropic * eta_mechanical

    return {
        'fluid': fluid,
        'inputs': dict(T_in_C=T_in_C, P_in_bar=P_in_bar, P_out_bar=P_out_bar,
                       m_dot=m_dot, eta_isentropic=eta_isentropic,
                       eta_mechanical=eta_mechanical),
        's1': dict(T=T_in,  P=P_in,  h=h1,  s=s1, rho=rho1, phase=phase_in),
        's2s': dict(T=T2s,  P=P_out, h=h2s, s=s1),
        's2': dict(T=T2,    P=P_out, h=h2,  s=s2, phase=phase_out, x=x2),
        'perf': dict(dh_isen=dh_isen, dh_act=dh_act,
                     W_isen=W_isen, W_shaft=W_shaft, W_out=W_out,
                     eta_overall=eta_overall),
    }


def print_report(res):
    inp, s1, s2s, s2, p = (res['inputs'], res['s1'], res['s2s'],
                           res['s2'], res['perf'])
    line = '-' * 60
    print(line)
    print(f" TURBINE PERFORMANCE REPORT  ({res['fluid']})")
    print(line)
    print(" Inputs")
    print(f"   Inlet temperature       : {inp['T_in_C']:>10.2f}  degC")
    print(f"   Inlet pressure          : {inp['P_in_bar']:>10.2f}  bar")
    print(f"   Outlet pressure         : {inp['P_out_bar']:>10.2f}  bar")
    print(f"   Mass flow rate          : {inp['m_dot']:>10.3f}  kg/s")
    print(f"   Isentropic efficiency   : {inp['eta_isentropic']:>10.3f}")
    print(f"   Mechanical efficiency   : {inp['eta_mechanical']:>10.3f}")
    print(line)
    print(" State 1 - Inlet")
    print(f"   T = {s1['T']-273.15:>8.2f} degC    P = {s1['P']/1e5:>8.2f} bar")
    print(f"   h = {s1['h']/1e3:>8.2f} kJ/kg   s = {s1['s']/1e3:>8.4f} kJ/kgK")
    print(f"   rho = {s1['rho']:>6.3f} kg/m3    phase: {s1['phase']}")
    print(" State 2s - Isentropic (ideal) outlet")
    print(f"   T = {s2s['T']-273.15:>8.2f} degC    P = {s2s['P']/1e5:>8.2f} bar")
    print(f"   h = {s2s['h']/1e3:>8.2f} kJ/kg")
    print(" State 2  - Actual outlet")
    print(f"   T = {s2['T']-273.15:>8.2f} degC    P = {s2['P']/1e5:>8.2f} bar")
    print(f"   h = {s2['h']/1e3:>8.2f} kJ/kg   s = {s2['s']/1e3:>8.4f} kJ/kgK")
    print(f"   phase: {s2['phase']}", end='')
    if s2['x'] is not None and 0.0 <= s2['x'] <= 1.0:
        print(f"   quality x = {s2['x']:.4f}")
    else:
        print()
    print(line)
    print(" Performance")
    print(f"   Isentropic enthalpy drop : {p['dh_isen']/1e3:>10.2f} kJ/kg")
    print(f"   Actual    enthalpy drop  : {p['dh_act']/1e3:>10.2f} kJ/kg")
    print(f"   Ideal (isentropic) power : {p['W_isen']/1e3:>10.2f} kW")
    print(f"   Shaft power              : {p['W_shaft']/1e3:>10.2f} kW")
    print(f"   Output (electrical) POWER: {p['W_out']/1e3:>10.2f} kW")
    print(f"   Overall efficiency       : {p['eta_overall']:>10.4f}")
    print(line)


def plot_Ts(res):
    """T-s diagram of the expansion. Useful when the working fluid can enter
    the two-phase region (steam, organic Rankine fluids)."""
    fluid = res['fluid']
    s1, s2s, s2 = res['s1'], res['s2s'], res['s2']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Saturation dome (skip for non-condensing fluids like Air)
    try:
        T_tr  = PropsSI('Ttriple', fluid)
        T_cr  = PropsSI('Tcrit',   fluid)
        Ts = np.linspace(T_tr + 0.5, T_cr - 0.5, 200)
        s_liq = [PropsSI('S', 'T', T, 'Q', 0, fluid) / 1e3 for T in Ts]
        s_vap = [PropsSI('S', 'T', T, 'Q', 1, fluid) / 1e3 for T in Ts]
        ax.plot(s_liq, Ts - 273.15, 'k-', lw=1.2)
        ax.plot(s_vap, Ts - 273.15, 'k-', lw=1.2, label='Saturation dome')
    except Exception:
        pass

    # Isobars through inlet and outlet pressures
    for P, style, lbl in [(s1['P'], 'b--', f"{s1['P']/1e5:.2f} bar (inlet)"),
                          (s2['P'], 'r--', f"{s2['P']/1e5:.2f} bar (outlet)")]:
        try:
            T_iso = np.linspace(PropsSI('Ttriple', fluid) + 1,
                                min(PropsSI('Tcrit', fluid) * 1.6, 1500), 250)
            s_iso = []
            T_keep = []
            for T in T_iso:
                try:
                    s_iso.append(PropsSI('S', 'T', T, 'P', P, fluid) / 1e3)
                    T_keep.append(T - 273.15)
                except Exception:
                    pass
            ax.plot(s_iso, T_keep, style, lw=0.9, alpha=0.7, label=lbl)
        except Exception:
            pass

    # Process points
    s1k, T1c = s1['s']/1e3, s1['T']-273.15
    s2sk, T2sc = s2s['s']/1e3, s2s['T']-273.15
    s2k, T2c = s2['s']/1e3, s2['T']-273.15

    ax.plot([s1k, s2sk], [T1c, T2sc], 'g-',  lw=2, label='Isentropic 1->2s')
    ax.plot([s1k, s2k],  [T1c, T2c],  'r-',  lw=2, label='Actual    1->2')

    ax.scatter([s1k, s2sk, s2k], [T1c, T2sc, T2c],
               c=['blue', 'green', 'red'], zorder=5)
    ax.annotate('1 (inlet)',  (s1k, T1c),  textcoords='offset points',
                xytext=(8, 8))
    ax.annotate('2s (ideal)', (s2sk, T2sc), textcoords='offset points',
                xytext=(8, -14))
    ax.annotate('2 (actual)', (s2k, T2c),  textcoords='offset points',
                xytext=(8, 8))

    ax.set_xlabel('Specific entropy  s  [kJ/(kg·K)]')
    ax.set_ylabel('Temperature  T  [°C]')
    ax.set_title(f'Turbine expansion on T-s diagram  —  {fluid}')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig('Turbine_Ts_diagram.png', dpi=130)
    plt.show()


if __name__ == "__main__":
    result = turbine(fluid, T_in_C, P_in_bar, P_out_bar, m_dot,
                     eta_isentropic, eta_mechanical)
    print_report(result)
    plot_Ts(result)
