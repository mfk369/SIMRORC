"""
Author: fkarim
"""

from CoolProp.CoolProp import PropsSI, PhaseSI


# ======================================================================
#                           USER INPUTS
# ======================================================================
fluid          = 'Isobutane'    # CoolProp fluid: Water, Isobutane, R134a, R245fa, CO2, ...
T_in_C         = 40      # Inlet temperature   [degC]
P_in_bar       = 1        # Inlet pressure      [bar]
P_out_bar      = 33      # Target outlet press [bar]
m_dot          = 50.0       # Mass flow rate      [kg/s]
eta_pump       = 0.8       # Pump isentropic efficiency (0-1)
# ======================================================================


def pump(fluid, T_in_C, P_in_bar, P_out_bar, m_dot, eta_pump):
    """Compute pump states and required power.

    Equations
    ---------
        State 1 (inlet)              : h1 = h(T1,P1) ;  s1 = s(T1,P1)
        State 2s (isentropic outlet) : h2s = h(P2, s1)
        Pump isentropic efficiency   : eta_p = (h2s - h1) / (h2 - h1)
        Actual outlet enthalpy       : h2  = h1 + (h2s - h1) / eta_p
        Shaft (electrical) power     : W_shaft = m_dot * (h2 - h1)
        Hydraulic (ideal) power      : W_hyd  = m_dot * v1 * (P2 - P1)
    """
    T_in  = T_in_C + 273.15
    P_in  = P_in_bar * 1e5
    P_out = P_out_bar * 1e5

    if P_out <= P_in:
        raise ValueError("Outlet pressure must be higher than inlet pressure.")
    if not (0 < eta_pump <= 1):
        raise ValueError("Pump efficiency must be in (0, 1].")

    # State 1 - inlet
    h1   = PropsSI('H', 'T', T_in, 'P', P_in, fluid)
    s1   = PropsSI('S', 'T', T_in, 'P', P_in, fluid)
    rho1 = PropsSI('D', 'T', T_in, 'P', P_in, fluid)
    v1   = 1.0 / rho1
    phase_in = PhaseSI('T', T_in, 'P', P_in, fluid)

    # --- Inlet must be 100% liquid (otherwise the pump will cavitate) ----
    # Strategy: require P_in to be subcritical AND T_in < T_sat(P_in)
    # so the fluid is subcooled liquid. Reject vapour, two-phase, or
    # supercritical inlet states.
    P_crit = PropsSI('Pcrit', fluid)
    if P_in >= P_crit:
        raise ValueError(
            f"Inlet is not 100% liquid: inlet pressure {P_in_bar:.3f} bar "
            f">= critical pressure {P_crit/1e5:.3f} bar for {fluid}. "
            f"Pump requires a subcooled-liquid inlet.")
    T_sat_in = PropsSI('T', 'P', P_in, 'Q', 0, fluid)   # bubble-point T at P_in
    subcool_K = T_sat_in - T_in
    if subcool_K <= 0.0:
        raise ValueError(
            f"Inlet is not 100% liquid for {fluid}: T_in = {T_in_C:.2f} degC "
            f"is at/above saturation T_sat(P_in) = {T_sat_in-273.15:.2f} degC "
            f"at {P_in_bar:.3f} bar (phase reported: '{phase_in}'). "
            f"Increase inlet pressure or lower inlet temperature so the "
            f"fluid is subcooled liquid.")
    if phase_in.lower() != 'liquid':
        raise ValueError(
            f"Inlet phase reported by CoolProp is '{phase_in}', not 'liquid'. "
            f"Pump requires a 100% liquid inlet.")

    # State 2s - ideal isentropic outlet (s2s = s1)
    h2s = PropsSI('H', 'P', P_out, 'S', s1, fluid)
    T2s = PropsSI('T', 'P', P_out, 'S', s1, fluid)

    # State 2 - actual outlet (pump-efficiency definition)
    dh_isen = h2s - h1
    dh_act  = dh_isen / eta_pump
    h2 = h1 + dh_act
    T2 = PropsSI('T', 'P', P_out, 'H', h2, fluid)
    s2 = PropsSI('S', 'P', P_out, 'H', h2, fluid)
    phase_out = PhaseSI('P', P_out, 'H', h2, fluid)

    # Powers
    W_hyd   = m_dot * v1 * (P_out - P_in)   # incompressible-fluid ideal
    W_isen  = m_dot * dh_isen               # isentropic shaft power
    W_shaft = m_dot * dh_act                # actual shaft power required

    # Temperature rise across the pump
    dT = T2 - T_in

    return {
        'fluid': fluid,
        'inputs': dict(T_in_C=T_in_C, P_in_bar=P_in_bar, P_out_bar=P_out_bar,
                       m_dot=m_dot, eta_pump=eta_pump),
        's1':  dict(T=T_in,  P=P_in,  h=h1,  s=s1, rho=rho1, v=v1,
                    phase=phase_in, T_sat=T_sat_in, subcool_K=subcool_K),
        's2s': dict(T=T2s,   P=P_out, h=h2s, s=s1),
        's2':  dict(T=T2,    P=P_out, h=h2,  s=s2, phase=phase_out),
        'perf': dict(dh_isen=dh_isen, dh_act=dh_act,
                     W_hyd=W_hyd, W_isen=W_isen, W_shaft=W_shaft,
                     dT=dT),
    }


def print_report(res):
    inp, s1, s2s, s2, p = (res['inputs'], res['s1'], res['s2s'],
                           res['s2'], res['perf'])
    line = '-' * 60
    print(line)
    print(f" PUMP PERFORMANCE REPORT  ({res['fluid']})")
    print(line)
    print(" Inputs")
    print(f"   Inlet temperature        : {inp['T_in_C']:>10.2f}  degC")
    print(f"   Inlet pressure           : {inp['P_in_bar']:>10.2f}  bar")
    print(f"   Target outlet pressure   : {inp['P_out_bar']:>10.2f}  bar")
    print(f"   Mass flow rate           : {inp['m_dot']:>10.3f}  kg/s")
    print(f"   Pump efficiency          : {inp['eta_pump']:>10.3f}")
    print(line)
    print(" State 1 - Inlet")
    print(f"   T = {s1['T']-273.15:>8.2f} degC    P = {s1['P']/1e5:>8.2f} bar")
    print(f"   h = {s1['h']/1e3:>8.3f} kJ/kg   s = {s1['s']/1e3:>8.5f} kJ/kgK")
    print(f"   rho = {s1['rho']:>7.2f} kg/m3   v = {s1['v']*1e3:>7.4f} L/kg")
    print(f"   phase: {s1['phase']}   "
          f"T_sat(P_in) = {s1['T_sat']-273.15:.2f} degC   "
          f"subcooling = {s1['subcool_K']:.2f} K")
    print(" State 2s - Isentropic (ideal) outlet")
    print(f"   T = {s2s['T']-273.15:>8.2f} degC    P = {s2s['P']/1e5:>8.2f} bar")
    print(f"   h = {s2s['h']/1e3:>8.3f} kJ/kg")
    print(" State 2  - Actual outlet")
    print(f"   T = {s2['T']-273.15:>8.2f} degC    P = {s2['P']/1e5:>8.2f} bar")
    print(f"   h = {s2['h']/1e3:>8.3f} kJ/kg   s = {s2['s']/1e3:>8.5f} kJ/kgK")
    print(f"   phase: {s2['phase']}")
    print(line)
    print(" Performance")
    print(f"   Hydraulic ideal power    (v.dP)     : {p['W_hyd']/1e3:>10.3f} kW")
    print(f"   Isentropic shaft power   (m.dh_s)   : {p['W_isen']/1e3:>10.3f} kW")
    print(f"   ACTUAL shaft power REQUIRED         : {p['W_shaft']/1e3:>10.3f} kW")
    print(f"   Isentropic enthalpy rise            : {p['dh_isen']/1e3:>10.4f} kJ/kg")
    print(f"   Actual    enthalpy rise             : {p['dh_act']/1e3:>10.4f} kJ/kg")
    print(f"   Temperature rise across pump        : {p['dT']:>10.3f} K")
    print(line)


if __name__ == "__main__":
    result = pump(fluid, T_in_C, P_in_bar, P_out_bar, m_dot, eta_pump)
    print_report(result)
