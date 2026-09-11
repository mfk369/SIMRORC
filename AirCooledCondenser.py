
"""
Air-cooled condenser (induced-draft) - digital-twin model.

Three-zone formulation (desuperheat / condense / subcool) with:
  * proper LMTD per zone (no 2nd-law-violating subcool from a lumped model),
  * a real minimum-approach (pinch) check,
  * a fouling resistance input so the twin can track degradation of U,
  * a rating-mode solver: given A, U, R_foul, ambient and inlet streams,
    return the heat duty, outlet state and fan power.

Energy balance (steady, counter-flow):
    Q = m_dot * (h_in - h_out) = m_dot_air * cp_air * (T_air_out - T_air_in)
Zone split using h_sv, h_sl at the condensing pressure:
    Q1 = desuperheat   (vapour, T_in -> T_cond)
    Q2 = condense      (two-phase, isothermal at T_cond)
    Q3 = subcool       (liquid,  T_cond -> T_out)
Per zone (counter-flow LMTD):
    A_zone = Q_zone / (U_eff * LMTD_zone)
Effective overall coefficient (fouled):
    U_eff = 1 / (1/U_clean + R_foul)

Author: fkarim
"""

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI, PhaseSI


# ======================================================================
#                           USER INPUTS
# ======================================================================
# --- Hot organic stream (inlet) ---------------------------------------
fluid          = 'Isobutane'    # CoolProp fluid name
T_in_C         = 50         # Inlet temperature   [degC]
P_in_bar       = 5.0            # Inlet pressure      [bar]
m_dot          = 10.0           # Mass flow rate      [kg/s]

# --- Ambient air ------------------------------------------------------
T_air_in_C     = 25.0           # Air inlet (ambient) [degC]
P_atm_bar      = 1.01325        # Atmospheric press.  [bar]
# m_dot_air is NOT specified — the design routine solves for the air flow
# that gives a saturated-liquid outlet (no subcool) at the installed A & U.

# --- Heat exchanger -------------------------------------------------
A_m2           = 487.6  # Total finned-tube area [m2]
U_clean_W_m2K  = 850.0           # Clean overall U (air-side basis) [W/m2K]
R_foul_m2K_W   = 0.0            # Fouling resistance [m2K/W]; >0 degrades U
dT_min_pinch_K = 5.0            # Minimum approach allowed [K]; warn if below

# --- Induced-draft fans ----------------------------------------------
dP_air_Pa      = 150.0          # Air-side pressure drop [Pa]
eta_fan        = 1         # Fan total efficiency (0-1)

# --- Optional ---------------------------------------------------------
dP_process_bar = 0.0            # Process-side pressure drop [bar]
# ======================================================================


CP_AIR = 1005.0       # J/(kg.K) dry air at ambient
R_AIR  = 287.05       # J/(kg.K)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _lmtd(dT_h, dT_c):
    """Counter-flow LMTD. Returns None if infeasible (any approach <= 0)."""
    if dT_h <= 0 or dT_c <= 0:
        return None
    if abs(dT_h - dT_c) < 1e-9:
        return 0.5 * (dT_h + dT_c)
    return (dT_h - dT_c) / np.log(dT_h / dT_c)


def _zone_split(Q, h_in, h_sv, h_sl, m_dot):
    Q_dsh_max = max(m_dot * (h_in - h_sv), 0.0)
    Q_cnd_max = m_dot * (h_sv - h_sl)
    Q1 = min(Q, Q_dsh_max)
    Q2 = min(max(Q - Q1, 0.0), Q_cnd_max)
    Q3 = max(Q - Q1 - Q2, 0.0)
    return Q1, Q2, Q3


def _zone_solution(Q, *, fluid, m_dot, h_in, T_in, P_out,
                   h_sv, h_sl, T_cond, C_air, T_air_in, U_eff):
    """Given a guess for Q, return area required and useful intermediates.

    Returns dict with A_required (or np.inf if infeasible) and the full
    geometry of intermediate states needed for plotting & reporting.
    """
    Q1, Q2, Q3 = _zone_split(Q, h_in, h_sv, h_sl, m_dot)

    # Air-side intermediate temperatures (counter-flow: air enters at the
    # organic-outlet side, so it picks up Q3, then Q2, then Q1).
    T_a_in  = T_air_in
    T_a_b   = T_a_in + Q3 / C_air          # after subcool zone
    T_a_c   = T_a_b  + Q2 / C_air          # after condense zone
    T_a_out = T_a_c  + Q1 / C_air          # exits at hot end

    h_out = h_in - Q / m_dot
    T_out = PropsSI('T', 'P', P_out, 'H', h_out, fluid) if Q > 0 else T_in

    A_total = 0.0
    A_dsh = A_cnd = A_sub = 0.0
    LMTD_dsh = LMTD_cnd = LMTD_sub = None
    pinch_K = np.inf

    # --- Subcool zone (organic: T_cond -> T_out, air: T_a_b -> T_a_in) --
    if Q3 > 0:
        dT_h = T_cond - T_a_b     # hot-end of subcool (= organic at T_cond)
        dT_c = T_out  - T_a_in    # cold-end of subcool (organic outlet)
        lm = _lmtd(dT_h, dT_c)
        if lm is None:
            return dict(A_required=np.inf, feasible=False,
                        infeasible_zone='subcool',
                        dT_h=dT_h, dT_c=dT_c)
        LMTD_sub = lm
        A_sub = Q3 / (U_eff * lm)
        A_total += A_sub
        pinch_K = min(pinch_K, dT_h, dT_c)

    # --- Condense zone (organic: T_cond const, air: T_a_c -> T_a_b) -----
    if Q2 > 0:
        dT_h = T_cond - T_a_c
        dT_c = T_cond - T_a_b
        lm = _lmtd(dT_h, dT_c)
        if lm is None:
            return dict(A_required=np.inf, feasible=False,
                        infeasible_zone='condense',
                        dT_h=dT_h, dT_c=dT_c)
        LMTD_cnd = lm
        A_cnd = Q2 / (U_eff * lm)
        A_total += A_cnd
        pinch_K = min(pinch_K, dT_h, dT_c)

    # --- Desuperheat zone -----------------------------------------------
    if Q1 > 0:
        if Q2 == 0 and Q3 == 0:
            # No condensation has begun: outlet is superheated vapour at T_out.
            T_hot_cold_end = T_out
        else:
            T_hot_cold_end = T_cond
        dT_h = T_in - T_a_out
        dT_c = T_hot_cold_end - T_a_c
        lm = _lmtd(dT_h, dT_c)
        if lm is None:
            return dict(A_required=np.inf, feasible=False,
                        infeasible_zone='desuperheat',
                        dT_h=dT_h, dT_c=dT_c)
        LMTD_dsh = lm
        A_dsh = Q1 / (U_eff * lm)
        A_total += A_dsh
        pinch_K = min(pinch_K, dT_h, dT_c)

    return dict(
        A_required=A_total, feasible=True,
        Q1=Q1, Q2=Q2, Q3=Q3,
        A_dsh=A_dsh, A_cnd=A_cnd, A_sub=A_sub,
        LMTD_dsh=LMTD_dsh, LMTD_cnd=LMTD_cnd, LMTD_sub=LMTD_sub,
        T_a_in=T_a_in, T_a_b=T_a_b, T_a_c=T_a_c, T_a_out=T_a_out,
        h_out=h_out, T_out=T_out, pinch_K=pinch_K,
    )


def _solve_Q(A_target, **kw):
    """Bisection on Q in [eps, Q_max] until A_required == A_target."""
    m_dot, h_in, C_air, T_in, T_air_in = (kw['m_dot'], kw['h_in'],
                                          kw['C_air'], kw['T_in'],
                                          kw['T_air_in'])
    # Hard upper bound: thermo cap. Tightened to 0.999 to keep LMTD finite.
    Q_air_cap = C_air * (T_in - T_air_in)
    # Organic-side cap: would require organic to leave at ambient (h at ambient)
    h_floor   = PropsSI('H', 'T', T_air_in, 'P', kw['P_out'], kw['fluid'])
    Q_org_cap = m_dot * (h_in - h_floor)
    Q_max = 0.999 * min(Q_air_cap, Q_org_cap)
    if Q_max <= 0:
        return None, 'no-driving-force'

    lo, hi = 1.0, Q_max
    f_lo = _zone_solution(lo, **kw)['A_required'] - A_target
    f_hi = _zone_solution(hi, **kw)['A_required'] - A_target
    if f_lo > 0:
        return None, 'A-too-small'   # even at Q->0 we need less area than asked
    if f_hi < 0:
        # Even the thermo cap doesn't need this much area -> air/area-limited
        return Q_max, 'air-limited'

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = _zone_solution(mid, **kw)['A_required'] - A_target
        if abs(f_mid) < 1e-3 or (hi - lo) < 1e-4:
            return mid, 'converged'
        if f_mid < 0:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi), 'max-iter'


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def air_cooled_condenser(*, fluid, T_in_C, P_in_bar, m_dot,
                         T_air_in_C, m_dot_air, P_atm_bar,
                         A_m2, U_clean_W_m2K, R_foul_m2K_W=0.0,
                         dT_min_pinch_K=5.0,
                         dP_air_Pa=150.0, eta_fan=0.65,
                         dP_process_bar=0.0):

    T_in  = T_in_C   + 273.15
    T_air = T_air_in_C + 273.15
    P_in  = P_in_bar    * 1e5
    P_out = (P_in_bar - dP_process_bar) * 1e5
    P_atm = P_atm_bar   * 1e5

    if P_out <= 0:
        raise ValueError("Process-side dP leaves P_out <= 0.")

    # --- Inlet organic state ------------------------------------------
    h_in   = PropsSI('H', 'T', T_in, 'P', P_in, fluid)
    s_in   = PropsSI('S', 'T', T_in, 'P', P_in, fluid)
    rho_in = PropsSI('D', 'T', T_in, 'P', P_in, fluid)
    phase_in = PhaseSI('T', T_in, 'P', P_in, fluid)

    # --- Saturation properties at the condensing pressure -------------
    P_crit = PropsSI('Pcrit', fluid)
    if P_in >= P_crit:
        raise ValueError(
            f"P_in {P_in_bar:.2f} bar >= P_crit {P_crit/1e5:.2f} bar -> "
            f"no two-phase condensation for {fluid}.")
    T_cond = PropsSI('T', 'P', P_in, 'Q', 0, fluid)
    h_sv   = PropsSI('H', 'P', P_in, 'Q', 1, fluid)
    h_sl   = PropsSI('H', 'P', P_in, 'Q', 0, fluid)

    if T_air >= T_cond:
        raise ValueError(
            f"T_air_in {T_air_in_C:.2f} degC >= T_cond "
            f"{T_cond-273.15:.2f} degC; no condensation possible.")
    if T_in < T_cond - 1e-6:
        raise ValueError(
            f"Inlet T {T_in_C:.2f} degC < T_sat(P_in) "
            f"{T_cond-273.15:.2f} degC; stream is not vapour.")

    # --- U with fouling -----------------------------------------------
    U_eff = 1.0 / (1.0 / U_clean_W_m2K + R_foul_m2K_W)
    C_air = m_dot_air * CP_AIR

    kw = dict(fluid=fluid, m_dot=m_dot, h_in=h_in, T_in=T_in, P_out=P_out,
              h_sv=h_sv, h_sl=h_sl, T_cond=T_cond,
              C_air=C_air, T_air_in=T_air, U_eff=U_eff)

    # --- Solve for Q --------------------------------------------------
    Q_solution, status = _solve_Q(A_m2, **kw)
    if Q_solution is None:
        raise ValueError(f"No feasible operating point: {status}.")
    sol = _zone_solution(Q_solution, **kw)

    # --- Outlet organic state ----------------------------------------
    h_out = sol['h_out']
    T_out = sol['T_out']
    s_out = PropsSI('S', 'P', P_out, 'H', h_out, fluid)
    phase_out = PhaseSI('P', P_out, 'H', h_out, fluid)
    try:
        x_out = PropsSI('Q', 'P', P_out, 'H', h_out, fluid)
    except Exception:
        x_out = None
    if h_out > h_sv:
        regime = 'superheated vapour (no condensation reached)'
    elif h_out > h_sl + 1e-3:
        regime = 'two-phase (partial condensation)'
    elif abs(h_out - h_sl) <= 1e-3:
        regime = 'saturated liquid'
    else:
        regime = 'subcooled liquid'

    # --- Pinch check --------------------------------------------------
    pinch_OK = sol['pinch_K'] >= dT_min_pinch_K

    # --- Fan power ----------------------------------------------------
    rho_air = P_atm / (R_AIR * T_air)
    V_air   = m_dot_air / rho_air
    W_fan_hyd  = V_air * dP_air_Pa
    W_fan_elec = W_fan_hyd / eta_fan

    # --- UA breakdown (digital-twin calibration handles) --------------
    UA_total_eff = sol['A_required'] * U_eff if sol['A_required'] > 0 else 0.0
    UA_installed = A_m2 * U_eff

    return {
        'fluid': fluid, 'status': status,
        'inputs': dict(T_in_C=T_in_C, P_in_bar=P_in_bar, m_dot=m_dot,
                       T_air_in_C=T_air_in_C, m_dot_air=m_dot_air,
                       A_m2=A_m2, U_clean=U_clean_W_m2K,
                       R_foul=R_foul_m2K_W, U_eff=U_eff,
                       dT_min_pinch_K=dT_min_pinch_K,
                       dP_air_Pa=dP_air_Pa, eta_fan=eta_fan,
                       dP_process_bar=dP_process_bar, P_atm_bar=P_atm_bar),
        'sat': dict(T_cond=T_cond, h_sv=h_sv, h_sl=h_sl,
                    h_fg=h_sv-h_sl, P_crit=P_crit),
        's_in':  dict(T=T_in,  P=P_in,  h=h_in,  s=s_in,
                      rho=rho_in, phase=phase_in),
        's_out': dict(T=T_out, P=P_out, h=h_out, s=s_out,
                      phase=phase_out, x=x_out, regime=regime),
        'air':   dict(T_air_in=T_air, T_air_b=sol['T_a_b'],
                      T_air_c=sol['T_a_c'], T_air_out=sol['T_a_out'],
                      rho=rho_air, V_dot=V_air, C_air=C_air),
        'hx':    dict(Q=Q_solution,
                      Q_dsh=sol['Q1'], Q_cnd=sol['Q2'], Q_sub=sol['Q3'],
                      A_required=sol['A_required'], A_installed=A_m2,
                      A_dsh=sol['A_dsh'], A_cnd=sol['A_cnd'],
                      A_sub=sol['A_sub'],
                      LMTD_dsh=sol['LMTD_dsh'], LMTD_cnd=sol['LMTD_cnd'],
                      LMTD_sub=sol['LMTD_sub'],
                      UA_eff=UA_total_eff, UA_installed=UA_installed,
                      pinch_K=sol['pinch_K'], pinch_OK=pinch_OK),
        'fan':   dict(W_fan_hyd=W_fan_hyd, W_fan_elec=W_fan_elec),
    }


# ---------------------------------------------------------------------
# Design mode: solve for m_dot_air that gives a saturated-liquid outlet
# ---------------------------------------------------------------------
def air_cooled_condenser_design(*, fluid, T_in_C, P_in_bar, m_dot,
                                T_air_in_C, P_atm_bar,
                                A_m2, U_clean_W_m2K, R_foul_m2K_W=0.0,
                                dT_min_pinch_K=5.0,
                                dP_air_Pa=150.0, eta_fan=0.65,
                                dP_process_bar=0.0,
                                m_air_min=1.0, m_air_max=10000.0,
                                m_air_init=None, tol_kgs=0.5):
    """Design mode — outlet target: saturated liquid (x_out = 0, no subcool).

    Bisects m_dot_air until the rating-mode duty equals the target duty
    Q_target = m_dot * (h_in - h_sl). At that air flow the unit just
    finishes condensation; any more air would over-cool (subcool).

    Warm-start (for QSS time-marching, fault-tracking, sensitivity sweeps):
      * m_air_init : if given, narrow the bisection bracket around it
                     ([0.5*m_init, 2*m_init]). Falls back to the full
                     range if the narrow bracket does not contain the
                     root. Typical speed-up: 5-10x in a time-march where
                     consecutive timesteps have similar operating points.
      * tol_kgs    : bisection convergence tolerance on m_air [kg/s].
                     0.5 kg/s is far below any physically meaningful
                     resolution; the original 1e-3 was overkill.

    Returns the same dict as air_cooled_condenser() plus a 'design'
    sub-dict with the solved air flow.
    """
    T_in = T_in_C + 273.15
    P_in = P_in_bar * 1e5
    h_in = PropsSI('H', 'T', T_in, 'P', P_in, fluid)
    h_sl = PropsSI('H', 'P', P_in, 'Q', 0, fluid)
    Q_target = m_dot * (h_in - h_sl)

    def _rate(m_air):
        try:
            r = air_cooled_condenser(
                fluid=fluid, T_in_C=T_in_C, P_in_bar=P_in_bar, m_dot=m_dot,
                T_air_in_C=T_air_in_C, m_dot_air=m_air, P_atm_bar=P_atm_bar,
                A_m2=A_m2, U_clean_W_m2K=U_clean_W_m2K,
                R_foul_m2K_W=R_foul_m2K_W, dT_min_pinch_K=dT_min_pinch_K,
                dP_air_Pa=dP_air_Pa, eta_fan=eta_fan,
                dP_process_bar=dP_process_bar)
            return r['hx']['Q'], r
        except ValueError:
            return 0.0, None

    # Set the bisection bracket.  Warm-start from m_air_init if available.
    lo, hi = m_air_min, m_air_max
    if m_air_init is not None and m_air_init > 0:
        lo_n = max(0.5 * m_air_init, m_air_min)
        hi_n = min(2.0 * m_air_init, m_air_max)
        Q_lo_n, _ = _rate(lo_n)
        Q_hi_n, _ = _rate(hi_n)
        if Q_lo_n <= Q_target <= Q_hi_n:
            lo, hi = lo_n, hi_n            # narrow bracket valid - use it
        else:
            # narrow bracket missed the root - validate the wide bracket
            Q_hi, _ = _rate(m_air_max)
            if Q_hi < Q_target * 0.999:
                raise ValueError(
                    f"Even at m_dot_air={m_air_max} kg/s the bundle cannot "
                    f"fully condense the stream. Increase A or U, or lower "
                    f"ambient T. (Q_max={Q_hi/1e3:.1f} kW vs target "
                    f"{Q_target/1e3:.1f} kW)")
    else:
        Q_hi, _ = _rate(m_air_max)
        if Q_hi < Q_target * 0.999:
            raise ValueError(
                f"Even at m_dot_air={m_air_max} kg/s the bundle cannot fully "
                f"condense the stream. Increase A or U, or lower ambient T. "
                f"(Q_max={Q_hi/1e3:.1f} kW vs target {Q_target/1e3:.1f} kW)")

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        Q_mid, _ = _rate(mid)
        if Q_mid < Q_target:
            lo = mid          # not enough air — under-condensing
        else:
            hi = mid          # too much air — subcooling
        if (hi - lo) < tol_kgs:
            break

    m_air_design = 0.5 * (lo + hi)
    _, r_final = _rate(m_air_design)
    r_final['design'] = dict(
        mode='saturated-liquid outlet (no subcool)',
        m_dot_air_solved=m_air_design,
        Q_target=Q_target,
    )
    return r_final


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def print_report(r):
    inp, sat, sin, sout = r['inputs'], r['sat'], r['s_in'], r['s_out']
    air, hx, fan = r['air'], r['hx'], r['fan']
    line = '-' * 66
    print(line)
    print(f" AIR-COOLED CONDENSER (DIGITAL-TWIN MODEL)   fluid: {r['fluid']}")
    print(line)
    print(" Inputs")
    print(f"   Organic in : T={inp['T_in_C']:.2f} degC  "
          f"P={inp['P_in_bar']:.2f} bar  m={inp['m_dot']:.3f} kg/s")
    print(f"   Air in     : T={inp['T_air_in_C']:.2f} degC  "
          f"m_air={inp['m_dot_air']:.1f} kg/s")
    print(f"   HX         : A={inp['A_m2']:.0f} m2  "
          f"U_clean={inp['U_clean']:.2f}  R_foul={inp['R_foul']:.5f}  "
          f"-> U_eff={inp['U_eff']:.2f} W/m2K")
    print(f"   Fans       : dP={inp['dP_air_Pa']:.0f} Pa  "
          f"eta_fan={inp['eta_fan']:.2f}")
    print(line)
    print(" Saturation at condensing pressure")
    print(f"   T_cond = {sat['T_cond']-273.15:.2f} degC   "
          f"h_fg = {sat['h_fg']/1e3:.2f} kJ/kg")
    print(line)
    print(" Heat-transfer solution (3-zone, counter-flow LMTD)")
    print(f"   solver status     : {r['status']}")
    print(f"   Heat duty Q       : {hx['Q']/1e3:>10.2f} kW")
    print(f"     desuperheat Q1  : {hx['Q_dsh']/1e3:>10.2f} kW")
    print(f"     condense    Q2  : {hx['Q_cnd']/1e3:>10.2f} kW")
    print(f"     subcool     Q3  : {hx['Q_sub']/1e3:>10.2f} kW")
    print(f"   Area split (A_req should equal A_installed):")
    print(f"     A_dsh : {hx['A_dsh']:>8.1f} m2   "
          f"LMTD_dsh: {hx['LMTD_dsh'] or 0:.2f} K")
    print(f"     A_cnd : {hx['A_cnd']:>8.1f} m2   "
          f"LMTD_cnd: {hx['LMTD_cnd'] or 0:.2f} K")
    print(f"     A_sub : {hx['A_sub']:>8.1f} m2   "
          f"LMTD_sub: {hx['LMTD_sub'] or 0:.2f} K")
    print(f"     SUM   : {hx['A_required']:>8.1f} m2   "
          f"(installed = {hx['A_installed']:.1f} m2)")
    print(f"   UA effective      : {hx['UA_eff']/1e3:>10.2f} kW/K")
    print(f"   Min approach (pinch): {hx['pinch_K']:.2f} K  "
          f"({'OK' if hx['pinch_OK'] else 'BELOW TARGET '+str(inp['dT_min_pinch_K'])+' K!'})")
    print(line)
    print(" Outlet organic stream")
    print(f"   T_out  = {sout['T']-273.15:8.2f} degC   "
          f"P_out = {sout['P']/1e5:.3f} bar")
    print(f"   h_out  = {sout['h']/1e3:8.2f} kJ/kg   "
          f"s_out = {sout['s']/1e3:.4f} kJ/kgK")
    print(f"   phase  : {sout['phase']}", end='')
    if sout['x'] is not None and 0 <= sout['x'] <= 1:
        print(f"   quality x = {sout['x']:.4f}")
    else:
        print()
    print(f"   regime : {sout['regime']}")
    print(line)
    print(" Air-side temperatures (cold -> hot end)")
    print(f"   T_air_in  = {air['T_air_in']-273.15:.2f} degC")
    print(f"   after subcool : {air['T_air_b']-273.15:.2f} degC")
    print(f"   after condense: {air['T_air_c']-273.15:.2f} degC")
    print(f"   T_air_out = {air['T_air_out']-273.15:.2f} degC   "
          f"(total rise {air['T_air_out']-air['T_air_in']:.2f} K)")
    print(f"   V_dot_air = {air['V_dot']:.1f} m3/s   "
          f"rho={air['rho']:.3f} kg/m3")
    print(line)
    print(" Energy required (fan work only - heat is rejected, not consumed)")
    print(f"   Hydraulic fan power : {fan['W_fan_hyd']/1e3:>8.2f} kW")
    print(f"   ELECTRICAL FAN POWER: {fan['W_fan_elec']/1e3:>8.2f} kW")
    print(line)


# ---------------------------------------------------------------------
# T-Q composite plot
# ---------------------------------------------------------------------
def plot_TQ(r):
    sat, hx, sin, sout, air = r['sat'], r['hx'], r['s_in'], r['s_out'], r['air']
    m_dot = r['inputs']['m_dot']

    Q1, Q2, Q3 = hx['Q_dsh'], hx['Q_cnd'], hx['Q_sub']
    Q_total = hx['Q']

    # Organic side cumulative Q at the four breakpoints
    Q_breaks = [0.0, Q1, Q1+Q2, Q1+Q2+Q3]
    if Q1 > 0 and Q2 == 0 and Q3 == 0:
        # desuperheating only, ended above T_cond
        T_breaks = [sin['T']-273.15, sout['T']-273.15, sout['T']-273.15,
                    sout['T']-273.15]
    else:
        T_breaks = [sin['T']-273.15,
                    sat['T_cond']-273.15 if Q1 > 0 else sin['T']-273.15,
                    sat['T_cond']-273.15,
                    sout['T']-273.15]

    # Air side intermediate temps (counter-flow). x-axis goes 0->Q_total in
    # the organic flow direction; air's local T at organic-x is its T at the
    # same cross-section, which is (T_air_out at x=0) -> (T_air_in at x=Q_total).
    Q_air = [0.0, Q1, Q1+Q2, Q1+Q2+Q3]
    T_air = [air['T_air_out']-273.15,
             air['T_air_c']-273.15,
             air['T_air_b']-273.15,
             air['T_air_in']-273.15]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(np.array(Q_breaks)/1e3, T_breaks, 'r-o', lw=2,
            label=f"{r['fluid']} (process)")
    ax.plot(np.array(Q_air)/1e3, T_air, 'b-s', lw=2,
            label='Air (counter-flow)')
    ax.axhline(sat['T_cond']-273.15, color='gray', ls=':', lw=1,
               label=f"T_cond = {sat['T_cond']-273.15:.1f} °C")

    # Shade zone boundaries
    if Q1 > 0:
        ax.axvspan(0, Q1/1e3, alpha=0.06, color='orange')
        ax.text(Q1/2e3, max(T_breaks)*0.97, 'desup.',
                ha='center', fontsize=9, color='orange')
    if Q2 > 0:
        ax.axvspan(Q1/1e3, (Q1+Q2)/1e3, alpha=0.06, color='red')
        ax.text((Q1+Q2/2)/1e3, sat['T_cond']-273.15-2, 'condense',
                ha='center', fontsize=9, color='red')
    if Q3 > 0:
        ax.axvspan((Q1+Q2)/1e3, (Q1+Q2+Q3)/1e3, alpha=0.06, color='blue')
        ax.text((Q1+Q2+Q3/2)/1e3, T_breaks[-1]+1, 'subcool',
                ha='center', fontsize=9, color='blue')

    ax.set_xlabel('Cumulative heat transferred  Q  [kW]')
    ax.set_ylabel('Temperature  [°C]')
    pinch_msg = f"min approach = {hx['pinch_K']:.2f} K"
    ax.set_title(f"ACC T-Q diagram - 3-zone model ({r['fluid']})   {pinch_msg}")
    ax.grid(alpha=0.3); ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('AirCooledCondenser_TQ.png', dpi=130)
    plt.show()


# ---------------------------------------------------------------------
# Digital-twin calibration helper
# ---------------------------------------------------------------------
def calibrate_R_foul(measured_Q_kW, *, fluid, T_in_C, P_in_bar, m_dot,
                     T_air_in_C, m_dot_air, P_atm_bar,
                     A_m2, U_clean_W_m2K, **kwargs):
    """Back-out the fouling resistance that matches a measured heat duty.

    Useful for the digital twin: feed measured Q (or measured T_air_out and
    compute Q from it) at a known operating point, get the current R_foul.
    Track that over time to see how the bundle is ageing.
    """
    target = measured_Q_kW * 1e3
    lo, hi = 0.0, 1.0   # m2K/W; 1.0 is huge, plenty of headroom
    for _ in range(80):
        mid = 0.5*(lo+hi)
        try:
            r = air_cooled_condenser(
                fluid=fluid, T_in_C=T_in_C, P_in_bar=P_in_bar, m_dot=m_dot,
                T_air_in_C=T_air_in_C, m_dot_air=m_dot_air,
                P_atm_bar=P_atm_bar, A_m2=A_m2,
                U_clean_W_m2K=U_clean_W_m2K, R_foul_m2K_W=mid, **kwargs)
            Q = r['hx']['Q']
        except ValueError:
            Q = 0.0
        if Q > target:
            lo = mid   # not enough fouling, U is still too high
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return 0.5*(lo+hi)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    res = air_cooled_condenser_design(
        fluid=fluid, T_in_C=T_in_C, P_in_bar=P_in_bar, m_dot=m_dot,
        T_air_in_C=T_air_in_C, P_atm_bar=P_atm_bar,
        A_m2=A_m2, U_clean_W_m2K=U_clean_W_m2K,
        R_foul_m2K_W=R_foul_m2K_W,
        dT_min_pinch_K=dT_min_pinch_K,
        dP_air_Pa=dP_air_Pa, eta_fan=eta_fan,
        dP_process_bar=dP_process_bar,
    )
    print(f"\n>>> DESIGN MODE: solved m_dot_air = "
          f"{res['design']['m_dot_air_solved']:.2f} kg/s "
          f"for {res['design']['mode']}\n")
    print_report(res)
    plot_TQ(res)
