"""
Recuperated subcritical Organic Rankine Cycle (ORC) plant model - DESIGN MODE -
for an geothermal heat source.

Plant layout (all heat exchangers counter-flow)

   brine (geofluid):
      PRODUCTION (P_geo) ─► EVAPORATOR ─► PREHEATER ─► [injection pump to P_inj] ─► INJECTION WELL
                            (boiling)      (liquid -> bubble point)
      the brine loses dP_geo_hx_bar in each exchanger (0.3 bar by default); the injection
      pump at the reinjection well raises it from the resulting suction pressure to P_inj
      (EGS: P_inj well above P_geo to cover fracture / wellbore losses)

   working fluid:
      (1) condenser outlet ─► PUMP ─► (2) ─► RECUPERATOR cold side ─► (2r) ─► PREHEATER
      ─► (2p, saturated liquid) ─► EVAPORATOR ─► (3) ─► TURBINE ─► (4) ─► RECUPERATOR hot side
      ─► (4r) ─► AIR-COOLED CONDENSER ─► (1)

Working-fluid state points
   1   condenser outlet   saturated liquid at P_cond (optional subcooling)
   2   pump outlet        compressed liquid at P_evap
   2r  recuperator outlet preheated LIQUID at P_evap   (= preheater inlet)
   2p  preheater outlet   saturated liquid at P_evap   (= evaporator inlet)
   3   evaporator outlet  saturated (or superheated) vapour at P_evap (= turbine inlet)
   4   turbine outlet     superheated vapour at P_cond
   4r  recuperator outlet cooled vapour at P_cond      (= condenser inlet)

Component models reused from this folder
   Turbine.py             turbine()                       isentropic + mechanical efficiency
   Pump.py                pump()                          isentropic efficiency, liquid-inlet check
   HEX.py                 hex_rate(), _tq_profile()       counter-flow T-Q profile, pinch, area
   AirCooledCondenser.py  air_cooled_condenser(),
                          _zone_solution()                3-zone ACC (desuperheat/condense/subcool), fan power

Design rules
   * a minimum approach (pinch) in EVERY exchanger - separate values for the
     evaporator, preheater, recuperator and condenser - checked on the full T-Q profile,
   * the preheater brings the working fluid exactly to its bubble point; the
     evaporator only boils it (plus optional superheat),
   * NO VAPOUR BEFORE THE PREHEATER: state 2r is liquid with at least
     `dT_liquid_min_K` of subcooling; the recuperator duty is capped to guarantee it,
   * reinjection temperature = target +/- tolerance ("band" mode): the brine is
     never cooled below target - tol, and a design point whose pinch cannot bring
     it down to target + tol is rejected; "free" = pinch only; "fixed" = tol 0,
   * recuperator sized for maximum recovery allowed by the pinch and the caps,
   * condenser air flow = the smallest flow that respects the pinch,
   * brine pressure drop per exchanger is restored by the brine pump (auxiliary load).

Free design variables optimised for maximum NET electric power
   T_evap  evaporation (saturation) temperature
   T_cond  condensing (saturation) temperature
   dT_sh   superheat at turbine inlet (only if superheat_mode == "optimise")

Usage
   python ORC_recuperated.py                        optimise (band 60 +/- 10 degC, recuperated), report, plots
   python ORC_recuperated.py --geo-outlet free      pinch-limited outlet, no reinjection constraint
   python ORC_recuperated.py --T-geo-out 60 --T-geo-out-tol 0     force the injection temperature
   python ORC_recuperated.py --superheat optimise   let the optimiser use superheat
   python ORC_recuperated.py --no-recuperator       simple ORC for comparison
   python ORC_recuperated.py --compare              4-way table: fixed/free x recuperated/simple
   python ORC_recuperated.py --T-evap 65 --T-cond 43               evaluate one design point
   python ORC_recuperated.py --json out.json        dump the design summary
   streamlit run ORC_design_app.py                  graphical design interface (same model)

Author: fkarim (plant model), composing the component modules above.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, replace

import numpy as np
from scipy.optimize import brentq, minimize
from CoolProp.CoolProp import PropsSI

from Turbine import turbine
from Pump import pump
from HEX import hex_rate, _tq_profile
from AirCooledCondenser import air_cooled_condenser, _zone_solution, CP_AIR, R_AIR


# ---------------------------------------------------------------------
# numerical margins (the component models take (T, P) inputs; a state that
# sits exactly on the saturation line is ambiguous for CoolProp)
# ---------------------------------------------------------------------
PUMP_INLET_MARGIN_K = 0.01   # Pump.py rejects a saturated inlet -> 0.01 K numerical subcooling
SAT_VAPOUR_MARGIN_K = 0.10   # turbine inlet: 0.1 K above T_sat == "saturated vapour"
PINCH_TOL_K         = 0.02   # accepted deviation from the pinch target

# ORC working fluids that CoolProp knows and that are dry/isentropic enough
# for a single-stage expander (first entry is the default)
WORKING_FLUIDS = ["Isobutane", "n-Butane", "Isopentane", "n-Pentane", "R245fa",
                  "R1233zd(E)", "R1234ze(Z)", "R236fa", "R365mfc", "R134a",
                  "R1234yf", "n-Propane", "Cyclopentane", "MM"]
SUPERHEAT_MODES = ("none", "optimise", "fixed")
STATE_LABELS = ["1", "2", "2r", "2p", "3", "4", "4r"]


class Infeasible(Exception):
    """Design point violates a constraint. `violation` (K) grades the penalty,
    `info` carries diagnostics (e.g. the achievable outlet temperature)."""

    def __init__(self, msg: str, violation: float = 1.0, **info):
        super().__init__(msg)
        self.violation = max(float(violation), 1e-3)
        self.info = info


# =====================================================================
#                              INPUTS
# =====================================================================
@dataclass
class PlantSpec:
    # --- brine / geofluid (production -> injection) ---------------
    geo_fluid: str = "Water"
    T_geo_in_C: float = 100.0      # production temperature      [degC]
    P_geo_bar: float = 5.0         # production wellhead pressure = brine pressure at the evaporator inlet [bar]
    m_geo: float = 50.0            # brine mass flow             [kg/s]
    T_geo_out_C: float = 60.0      # reinjection target          [degC]
    T_geo_out_tol_K: float = 10.0  # allowed deviation +/-       [K]
    geo_outlet: str = "band"       # "band": target +/- tol | "free": pinch only | "fixed": band, tol 0
    dP_geo_hx_bar: float = 0.3     # brine pressure drop in EACH exchanger (evaporator, preheater) [bar]
    P_inj_bar: float = 5.0         # injection pump discharge pressure into the injection well [bar]
                                   # (EGS: set well above P_geo to cover fracture / wellbore losses)
    eta_geo_pump: float = 0.75     # injection (brine) pump efficiency
    eta_geo_motor: float = 0.95    # injection pump motor efficiency
    # --- working fluid / cycle -----------------------------------------
    wf: str = "Isobutane"
    recuperator: bool = True
    dT_subcool_K: float = 0.0      # condensate subcooling       [K]
    dT_liquid_min_K: float = 2.0   # min subcooling at preheater inlet (no vapour before preheater) [K]
    superheat_mode: str = "none"   # "none": saturated vapour | "optimise" | "fixed"
    dT_sh_fixed_K: float = 0.0     # superheat used when superheat_mode == "fixed" [K]
    pinch_evap_K: float = 10.0     # minimum approach, evaporator  [K]
    pinch_pre_K: float = 10.0      # minimum approach, preheater   [K]
    pinch_rec_K: float = 10.0      # minimum approach, recuperator [K]
    pinch_cond_K: float = 10.0     # minimum approach, condenser   [K]
    # --- ambient / air-cooled condenser --------------------------------
    T_air_in_C: float = 25.0       # ambient dry-bulb            [degC]
    P_atm_bar: float = 1.01325
    dP_air_Pa: float = 150.0       # ACC air-side pressure drop  [Pa]
    eta_fan: float = 0.65          # fan total efficiency
    U_acc: float = 850.0           # ACC overall U (from AirCooledCondenser.py) [W/m2K]
    R_foul_acc: float = 0.0        # ACC fouling resistance      [m2K/W]
    # --- turbomachinery ------------------------------------------------
    eta_turb_is: float = 0.88      # turbine isentropic eff.   (Turbine.py)
    eta_mech: float = 0.98         # turbine mechanical eff.   (Turbine.py)
    eta_gen: float = 0.97          # generator eff.
    eta_pump: float = 0.80         # feed pump isentropic eff. (Pump.py)
    eta_motor: float = 0.95        # feed pump motor eff.
    # --- exchanger U values (area estimates only; no effect on power) --
    U_evap: float = 1000.0         # brine / boiling working fluid   [W/m2K]
    U_pre: float = 800.0           # brine / liquid working fluid    [W/m2K]
    U_rec: float = 300.0           # vapour / liquid                 [W/m2K]
    n_seg: int = 60                # T-Q discretisation per exchanger

    def outlet_band_K(self):
        """(floor, ceiling) of the allowed brine outlet temperature [K], or None (free)."""
        if self.geo_outlet == "free":
            return None
        tol = 0.0 if self.geo_outlet == "fixed" else max(self.T_geo_out_tol_K, 0.0)
        T = self.T_geo_out_C + 273.15
        return (T - tol, T + tol)

    def superheat_for_optimiser(self):
        """None -> superheat is a free variable; else the value to hold."""
        if self.superheat_mode == "optimise":
            return None
        if self.superheat_mode == "fixed":
            return max(self.dT_sh_fixed_K, SAT_VAPOUR_MARGIN_K)
        return SAT_VAPOUR_MARGIN_K


@dataclass
class DesignVars:
    T_evap_C: float
    dT_sh_K: float
    T_cond_C: float


def _sat_T(P: float, fluid: str) -> float:
    return PropsSI("T", "P", P, "Q", 0, fluid)


# =====================================================================
#                      COMPONENT DESIGN WRAPPERS
# =====================================================================
def _empty_recuperator(h2, h4, T2, T4):
    return dict(active=False, q=0.0, h2r=h2, h4r=h4, T2r=T2, T4r=T4,
                A_per_kg=0.0, min_dT=None, capped=False, cap_reason=None,
                effectiveness=0.0, tq_T_h=None, tq_T_c=None, tq_Q=None)


def recuperator_design(spec: PlantSpec, P_evap: float, P_cond: float,
                       h2: float, h4: float, h2r_cap: float | None = None,
                       cap_reason: str | None = None) -> dict:
    """Size the recuperator for maximum recovery at the pinch (per kg/s of wf).

    Hot side : turbine exhaust vapour  4 -> 4r  at P_cond
    Cold side: pump discharge liquid   2 -> 2r  at P_evap

    `h2r_cap` limits the cold outlet enthalpy (liquid margin at the preheater
    inlet and/or the reinjection ceiling through the preheater cold-end pinch).

    For a dry fluid the vapour side has the smaller heat-capacity rate, so the
    pinch sits at the cold end (T_4r = T_2 + pinch).  That closed form is tried
    first and checked on the full T-Q profile; otherwise the general solver
    hex_rate() (huge area -> pinch-limited) is used.
    """
    wf, pinch, U, n = spec.wf, spec.pinch_rec_K, spec.U_rec, spec.n_seg
    T2 = PropsSI("T", "H", h2, "P", P_evap, wf)
    T4 = PropsSI("T", "H", h4, "P", P_cond, wf)
    common = dict(h_ci=h2, h_hi=h4, P_h=P_cond, P_c=P_evap, m_h=1.0, m_c=1.0,
                  hot_fluid=wf, cold_fluid=wf, U=U, n_seg=n)

    if T4 <= T2 + pinch + 1e-6:
        return _empty_recuperator(h2, h4, T2, T4)         # no driving force

    # closed form: cold-end pinch
    h4r = PropsSI("H", "T", T2 + pinch, "P", P_cond, wf)
    h2r = h2 + (h4 - h4r)
    ok = True
    try:
        T2r = PropsSI("T", "H", h2r, "P", P_evap, wf)
        if T2r > T4 - pinch + 1e-6:
            ok = False
        else:
            Q, T_h, T_c, Q_arr, A, min_dT = _tq_profile(h2r, **common)
            ok = min_dT >= pinch - PINCH_TOL_K
    except Exception:
        ok = False

    if not ok:                                             # general pinch solver
        r = hex_rate(hot_fluid=wf, P_h_bar=P_cond / 1e5, m_h=1.0, h_h_in_Jkg=h4,
                     cold_fluid=wf, P_c_bar=P_evap / 1e5, m_c=1.0, h_c_in_Jkg=h2,
                     A_m2=1e9, U_W_m2K=U, pinch_K=pinch, n_seg=n, name="Recuperator")
        h2r = r["h_c_out"]

    capped = False
    if h2r_cap is not None and h2r > h2r_cap:
        h2r, capped = h2r_cap, True

    if h2r - h2 < 1.0:                                     # < 1 J/kg -> nothing recovered
        out = _empty_recuperator(h2, h4, T2, T4)
        out.update(capped=capped, cap_reason=cap_reason if capped else None)
        return out

    Q, T_h, T_c, Q_arr, A, min_dT = _tq_profile(h2r, **common)
    h4r = h4 - Q
    q_max = h4 - PropsSI("H", "T", T2, "P", P_cond, wf)   # hot side cooled to T_2
    return dict(active=True, q=Q, h2r=h2r, h4r=h4r, T2r=float(T_c[-1]), T4r=float(T_h[0]),
                A_per_kg=A, min_dT=min_dT, capped=capped,
                cap_reason=cap_reason if capped else None,
                effectiveness=Q / q_max if q_max > 0 else 0.0,
                tq_T_h=T_h, tq_T_c=T_c, tq_Q=Q_arr)


def heat_input_design(spec: PlantSpec, P_evap: float, h2r: float, h3: float) -> dict:
    """Size the brine-side train (EVAPORATOR then PREHEATER, brine in series)
    and return the working-fluid flow.

    Evaporator: brine  T_geo_in (P_geo)      -> T_b1 (P_geo - dP)
                wf     2p saturated liquid   -> 3 (saturated / superheated vapour)
    Preheater : brine  T_b1 (P_geo - dP)     -> T_geo_out (P_geo - 2 dP)
                wf     2r liquid             -> 2p saturated liquid (bubble point)

    m_wf is the largest flow that keeps every pinch (evaporator and preheater,
    checked on the full profiles) without cooling the brine below the
    reinjection floor.  A design point whose pinch cannot bring the brine down
    to the reinjection ceiling raises Infeasible (achievable outlet in `info`).
    """
    wf, geo = spec.wf, spec.geo_fluid
    pe, pp = spec.pinch_evap_K, spec.pinch_pre_K
    m_w = spec.m_geo
    P_w0 = spec.P_geo_bar * 1e5
    P_w1 = P_w0 - spec.dP_geo_hx_bar * 1e5
    P_w2 = P_w1 - spec.dP_geo_hx_bar * 1e5
    if P_w2 <= 0.05e5:
        raise Infeasible("brine pressure drops exceed the loop pressure", 1.0)
    T_w_in = spec.T_geo_in_C + 273.15
    band = spec.outlet_band_K()
    T_floor = band[0] if band else -np.inf
    T_ceil = band[1] if band else np.inf
    T_sat_w = PropsSI("T", "P", P_w0, "Q", 0, geo)
    if T_w_in >= T_sat_w - 0.5:
        P_need = PropsSI("P", "T", T_w_in, "Q", 0, geo) / 1e5
        raise Infeasible(
            f"brine would flash: at {P_w0/1e5:.1f} bar {geo} boils at {T_sat_w-273.15:.1f} degC, below the "
            f"production temperature {T_w_in-273.15:.1f} degC; raise the brine pressure above {P_need:.1f} bar",
            1.0, P_geo_min_bar=P_need)
    hw_in = PropsSI("H", "T", T_w_in, "P", P_w0, geo)

    def hw(T, P):
        return PropsSI("H", "T", T, "P", P, geo)

    T2r = PropsSI("T", "H", h2r, "P", P_evap, wf)
    T3 = PropsSI("T", "H", h3, "P", P_evap, wf)
    T_evap = _sat_T(P_evap, wf)
    h_bub = PropsSI("H", "P", P_evap, "Q", 0, wf)
    h_dew = PropsSI("H", "P", P_evap, "Q", 1, wf)
    if T3 > T_w_in - pe + 1e-6:
        raise Infeasible("evaporator hot end: turbine inlet hotter than T_geo_in - pinch",
                         T3 - (T_w_in - pe))
    if h2r >= h_bub:
        raise Infeasible("preheater inlet is not liquid", 1.0)

    # candidate 1: evaporator cold end == preheater hot end (bubble point):
    #              T_b1 >= T_evap + max(pinch_evap, pinch_pre)
    T_b1_min = T_evap + max(pe, pp)
    if T_b1_min >= T_w_in - 0.01:
        raise Infeasible("evaporation temperature too close to the brine inlet", T_b1_min - T_w_in + 0.01)
    m_bub = m_w * (hw_in - hw(T_b1_min, P_w0)) / (h3 - h_bub)
    # candidate 2: preheater cold end (brine leaves at max(T_2r + pinch_pre, floor))
    T_out_min = max(T2r + pp, T_floor)
    if T_out_min >= T_w_in - 0.01:
        raise Infeasible("preheater cold end: wf inlet (or reinjection floor) too hot for the brine",
                         T_out_min - T_w_in + 0.01)
    m_cold = m_w * (hw_in - hw(T_out_min, P_w2)) / (h3 - h2r)
    m_wf = min(m_bub, m_cold)
    if m_bub <= m_cold:
        limiting = "bubble-point pinch (evaporator brine outlet)"
    elif T_floor > T2r + pp:
        limiting = "reinjection floor (target - tolerance)"
    else:
        limiting = "preheater cold-end pinch"

    ev_common = dict(h_ci=h_bub, h_hi=hw_in, P_h=P_w0, P_c=P_evap, m_h=m_w,
                     hot_fluid=geo, cold_fluid=wf, U=spec.U_evap, n_seg=spec.n_seg)
    pr_common = dict(h_ci=h2r, P_h=P_w1, P_c=P_evap, m_h=m_w,
                     hot_fluid=geo, cold_fluid=wf, U=spec.U_pre, n_seg=spec.n_seg)

    def profiles(m):
        ev = _tq_profile(h3, m_c=m, **ev_common)
        h_b1 = hw_in - ev[0] / m_w
        pr = _tq_profile(h_bub, m_c=m, h_hi=h_b1, **pr_common)
        return ev, pr, h_b1

    def slack(m):                       # >= 0 when both pinches hold; decreasing in m
        ev, pr, _ = profiles(m)
        return min(ev[5] - pe, pr[5] - pp)

    ev, pr, h_b1 = profiles(m_wf)
    sl = min(ev[5] - pe, pr[5] - pp)
    if sl < -PINCH_TOL_K:               # pinch inside a profile (cp(T) curvature) -> shrink m_wf
        lo = 0.5 * m_wf
        if slack(lo) < 0:
            lo = 0.05 * m_wf
        m_wf = brentq(slack, lo, m_wf, xtol=1e-5 * m_wf)
        ev, pr, h_b1 = profiles(m_wf)
        sl = min(ev[5] - pe, pr[5] - pp)
        limiting = "interior pinch (profile)"
    if sl < -PINCH_TOL_K:
        raise Infeasible("heat-input pinch cannot be met", -sl)

    Q_ev, T_h_ev, T_c_ev, Q_arr_ev, A_ev, dT_ev = ev
    Q_pr, T_h_pr, T_c_pr, Q_arr_pr, A_pr, dT_pr = pr
    h_out = h_b1 - Q_pr / m_w
    T_b1 = PropsSI("T", "H", h_b1, "P", P_w1, geo)
    T_out = PropsSI("T", "H", h_out, "P", P_w2, geo)
    if T_out >= PropsSI("T", "P", P_w2, "Q", 0, geo) - 0.5:
        raise Infeasible(
            f"brine would flash after the exchangers: {P_w2/1e5:.1f} bar at the preheater outlet is below "
            f"the boiling pressure at {T_out-273.15:.1f} degC; raise the production pressure", 1.0)
    if T_out > T_ceil + 1e-3:
        raise Infeasible(
            f"reinjection: the pinch only allows cooling the brine to {T_out-273.15:.1f} degC "
            f"at this design point, above the allowed {T_ceil-273.15:.1f} degC",
            T_out - T_ceil, T_geo_out_C=T_out - 273.15)

    # pinch locations
    i = int(np.argmin(T_h_ev - T_c_ev))
    h_c_i = h_bub + Q_arr_ev[i] / m_wf
    if i == 0:
        loc_ev = "brine outlet / bubble point (start of boiling)"
    elif i == len(T_h_ev) - 1:
        loc_ev = "hot end (brine inlet / turbine inlet)"
    elif abs(h_c_i - h_dew) <= 60.0:
        loc_ev = "dew point (end of boiling)"
    elif h_c_i < h_dew:
        loc_ev = "boiling zone"
    else:
        loc_ev = "superheating zone"
    j = int(np.argmin(T_h_pr - T_c_pr))
    if j == 0:
        loc_pr = "cold end (brine to injection / wf from recuperator)"
    elif j == len(T_h_pr) - 1:
        loc_pr = "hot end (bubble point)"
    else:
        loc_pr = "inside the preheater"

    evaporator = dict(Q=Q_ev, Q_boil=m_wf * (min(h_dew, h3) - h_bub), Q_sh=m_wf * max(h3 - h_dew, 0.0),
                      A=A_ev, min_dT=dT_ev, pinch_location=loc_ev,
                      pinch_Q_fraction=float(Q_arr_ev[i] / Q_ev), P_geo_in_bar=P_w0 / 1e5,
                      P_geo_out_bar=P_w1 / 1e5, T_geo_in=T_w_in, T_geo_out=T_b1,
                      tq_T_h=T_h_ev, tq_T_c=T_c_ev, tq_Q=Q_arr_ev)
    preheater = dict(Q=Q_pr, A=A_pr, min_dT=dT_pr, pinch_location=loc_pr,
                     pinch_Q_fraction=float(Q_arr_pr[j] / Q_pr) if Q_pr > 0 else 0.0,
                     P_geo_in_bar=P_w1 / 1e5, P_geo_out_bar=P_w2 / 1e5,
                     T_geo_in=T_b1, T_geo_out=T_out,
                     tq_T_h=T_h_pr, tq_T_c=T_c_pr, tq_Q=Q_arr_pr)
    brine_states = [
        dict(point="production (evaporator inlet)", T_C=T_w_in - 273.15, P_bar=P_w0 / 1e5, h_kJkg=hw_in / 1e3),
        dict(point="evaporator outlet / preheater inlet", T_C=T_b1 - 273.15, P_bar=P_w1 / 1e5, h_kJkg=h_b1 / 1e3),
        dict(point="preheater outlet / injection", T_C=T_out - 273.15, P_bar=P_w2 / 1e5, h_kJkg=h_out / 1e3),
    ]
    return dict(m_wf=m_wf, Q_total=Q_ev + Q_pr, limiting=limiting, evaporator=evaporator,
                preheater=preheater, brine_states=brine_states, hw_in=hw_in, h_out=h_out,
                T_geo_out=T_out, T_geo_mid=T_b1, T2r=T2r, T3=T3, h_bub=h_bub, h_dew=h_dew)


def brine_pump_design(spec: PlantSpec, T_suc: float, P_suc: float, h_suc: float) -> dict:
    """Injection pump at the reinjection well.

    Suction  : brine leaving the preheater (P_geo - 2 dP_hx, T_geo_out)
    Discharge: P_inj_bar (whatever the injection well / fractures need)
    The head is P_inj - P_suction; if P_inj is at or below the suction pressure
    no pump work is needed (head = 0, flagged).  The small temperature rise
    from pump inefficiency is carried into the injection-well state.
    """
    geo = spec.geo_fluid
    P_inj = spec.P_inj_bar * 1e5
    dP = max(P_inj - P_suc, 0.0)
    rho = PropsSI("D", "T", T_suc, "P", P_suc, geo)
    W_hyd = spec.m_geo / rho * dP
    W_shaft = W_hyd / spec.eta_geo_pump
    W_el = W_shaft / spec.eta_geo_motor
    h_inj = h_suc + W_shaft / spec.m_geo
    P_state = max(P_inj, P_suc)
    T_inj = PropsSI("T", "H", h_inj, "P", P_state, geo)
    return dict(P_suction_bar=P_suc / 1e5, P_inj_bar=P_state / 1e5, dP_bar=dP / 1e5,
                rho=rho, W_hyd=W_hyd, W_shaft=W_shaft, W_el=W_el,
                h_inj=h_inj, T_inj=T_inj, dT_K=T_inj - T_suc,
                no_head=(P_inj <= P_suc))


def condenser_design(spec: PlantSpec, P_cond: float, h4r: float, h1: float,
                     m_wf: float) -> dict:
    """Size the air-cooled condenser: smallest air flow that respects the pinch.

    Zones (counter-flow, air enters at the condensate end):
        desuperheat 4r -> dew, condense dew -> bubble, subcool bubble -> 1.
    Pinch candidates:
        condensing zone hot end  T_cond - T_air_after_condensing   (usually binding)
        desuperheat hot end      T_4r   - T_air_out
        subcool cold end         T_1    - T_air_in                 (flow independent)
    The area comes from AirCooledCondenser._zone_solution() (3-zone LMTD) and
    the fan power from the same volumetric-flow x dP / eta model.
    """
    wf, pinch = spec.wf, spec.pinch_cond_K
    T_air = spec.T_air_in_C + 273.15
    P_atm = spec.P_atm_bar * 1e5
    T_cond = _sat_T(P_cond, wf)
    h_sv = PropsSI("H", "P", P_cond, "Q", 1, wf)
    h_sl = PropsSI("H", "P", P_cond, "Q", 0, wf)
    T4r = PropsSI("T", "H", h4r, "P", P_cond, wf)
    T1 = PropsSI("T", "H", h1, "P", P_cond, wf)

    Q = m_wf * (h4r - h1)
    Q1 = m_wf * max(h4r - h_sv, 0.0)          # desuperheat
    Q3 = m_wf * max(h_sl - h1, 0.0)           # subcool
    Q2 = max(Q - Q1 - Q3, 0.0)                # condense

    if T1 - T_air < pinch - 1e-9:
        raise Infeasible("condenser cold end: T_1 - T_air < pinch", pinch - (T1 - T_air))
    d_cnd = T_cond - pinch - T_air
    if d_cnd <= 0:
        raise Infeasible("condenser: T_cond <= T_air + pinch", -d_cnd + 1e-3)
    C_air = (Q2 + Q3) / d_cnd
    d_dsh = T4r - pinch - T_air
    if d_dsh <= 0:
        raise Infeasible("condenser: T_4r <= T_air + pinch", -d_dsh + 1e-3)
    C_air = max(C_air, Q / d_dsh)
    m_air = C_air / CP_AIR

    U_eff = 1.0 / (1.0 / spec.U_acc + spec.R_foul_acc)
    z = _zone_solution(Q, fluid=wf, m_dot=m_wf, h_in=h4r, T_in=T4r, P_out=P_cond,
                       h_sv=h_sv, h_sl=h_sl, T_cond=T_cond, C_air=C_air,
                       T_air_in=T_air, U_eff=U_eff)
    if not z["feasible"]:
        raise Infeasible(f"condenser LMTD infeasible in {z['infeasible_zone']} zone", 1.0)
    if z["pinch_K"] < pinch - PINCH_TOL_K:
        raise Infeasible(f"condenser pinch {z['pinch_K']:.2f} K", pinch - z["pinch_K"])

    rho_air = P_atm / (R_AIR * T_air)
    V_air = m_air / rho_air
    W_fan = V_air * spec.dP_air_Pa / spec.eta_fan

    return dict(Q=Q, Q_dsh=z["Q1"], Q_cnd=z["Q2"], Q_sub=z["Q3"],
                A=z["A_required"], A_dsh=z["A_dsh"], A_cnd=z["A_cnd"], A_sub=z["A_sub"],
                LMTD_dsh=z["LMTD_dsh"], LMTD_cnd=z["LMTD_cnd"], LMTD_sub=z["LMTD_sub"],
                min_dT=z["pinch_K"], T_cond=T_cond, T4r=T4r, T1=T1,
                C_air=C_air, m_air=m_air, V_air=V_air, rho_air=rho_air, U_eff=U_eff,
                T_a_in=T_air, T_a_b=z["T_a_b"], T_a_c=z["T_a_c"], T_a_out=z["T_a_out"],
                dT_air=z["T_a_out"] - T_air, ITD=T_cond - T_air, W_fan=W_fan)


# =====================================================================
#                          CYCLE SOLVER
# =====================================================================
def solve_cycle(spec: PlantSpec, T_evap_C: float, dT_sh_K: float,
                T_cond_C: float) -> dict:
    """Solve the full recuperated ORC for one set of design variables.

    Returns a nested dict (states, components, powers, performance, checks).
    Raises Infeasible when a constraint cannot be met.
    """
    wf = spec.wf
    pe, pp, pc = spec.pinch_evap_K, spec.pinch_pre_K, spec.pinch_cond_K
    if T_cond_C < spec.T_air_in_C + pc:
        raise Infeasible("T_cond below T_air + condenser pinch", spec.T_air_in_C + pc - T_cond_C)
    if T_evap_C - T_cond_C < 5.0:
        raise Infeasible("T_evap too close to T_cond", 5.0 - (T_evap_C - T_cond_C))
    T_crit = PropsSI("Tcrit", wf)
    if T_evap_C + 273.15 >= T_crit - 1.0:
        raise Infeasible("T_evap at/above the critical point", 1.0)

    T_evap = T_evap_C + 273.15
    T_cond = T_cond_C + 273.15
    P_evap = PropsSI("P", "T", T_evap, "Q", 0, wf)
    P_cond = PropsSI("P", "T", T_cond, "Q", 0, wf)
    P_evap_bar, P_cond_bar = P_evap / 1e5, P_cond / 1e5

    # ---- 1 -> 2 feed pump (per kg/s) ----------------------------------
    T1_C = T_cond_C - max(spec.dT_subcool_K, PUMP_INLET_MARGIN_K)
    pm = pump(wf, T1_C, P_cond_bar, P_evap_bar, 1.0, spec.eta_pump)
    h1, s1, T1 = pm["s1"]["h"], pm["s1"]["s"], pm["s1"]["T"]
    h2, s2, T2 = pm["s2"]["h"], pm["s2"]["s"], pm["s2"]["T"]
    w_pump = pm["perf"]["dh_act"]                     # J/kg shaft

    # ---- 3 -> 4 turbine (per kg/s) -----------------------------------
    dT_sh_eff = max(dT_sh_K, SAT_VAPOUR_MARGIN_K)
    T3_C = T_evap_C + dT_sh_eff
    tb = turbine(wf, T3_C, P_evap_bar, P_cond_bar, 1.0, spec.eta_turb_is, spec.eta_mech)
    h3, s3, T3 = tb["s1"]["h"], tb["s1"]["s"], tb["s1"]["T"]
    h4, s4, T4 = tb["s2"]["h"], tb["s2"]["s"], tb["s2"]["T"]
    x4 = tb["s2"]["x"]
    w_turb = tb["perf"]["dh_act"]                     # J/kg shaft (before eta_mech)

    # ---- recuperator caps: 2r must stay liquid; band ceiling via preheater cold end
    dT_liq = max(spec.dT_liquid_min_K, 0.05)
    caps = [("liquid margin at preheater inlet", T_evap - dT_liq)]
    band = spec.outlet_band_K()
    if band is not None:
        caps.append(("reinjection ceiling minus preheater pinch", band[1] - pp))
    cap_reason, T_cap = min(caps, key=lambda c: c[1])
    h2r_cap = h2 if T_cap <= T2 else PropsSI("H", "T", T_cap, "P", P_evap, wf)

    # ---- 4 -> 4r / 2 -> 2r recuperator (per kg/s) --------------------
    if spec.recuperator:
        rec = recuperator_design(spec, P_evap, P_cond, h2, h4, h2r_cap, cap_reason)
    else:
        rec = _empty_recuperator(h2, h4, T2, T4)
    h2r, h4r = rec["h2r"], rec["h4r"]

    # ---- no vapour before the preheater --------------------------------
    subcool_2r = T_evap - rec["T2r"]
    if subcool_2r < dT_liq - 1e-3:
        raise Infeasible(f"preheater inlet subcooling {subcool_2r:.2f} K < {dT_liq:.2f} K",
                         dT_liq - subcool_2r)

    # ---- brine train: evaporator then preheater: sets m_wf -------------
    hi = heat_input_design(spec, P_evap, h2r, h3)
    m_wf = hi["m_wf"]
    ev, pr = hi["evaporator"], hi["preheater"]

    # ---- injection pump (at the reinjection well) -------------------------
    bp = brine_pump_design(spec, hi["T_geo_out"], pr["P_geo_out_bar"] * 1e5, hi["h_out"])
    brine_states = hi["brine_states"] + [
        dict(point="injection well (after injection pump)", T_C=bp["T_inj"] - 273.15,
             P_bar=bp["P_inj_bar"], h_kJkg=bp["h_inj"] / 1e3)]

    # ---- 4r -> 1 air-cooled condenser ---------------------------------
    cd = condenser_design(spec, P_cond, h4r, h1, m_wf)

    # ---- powers ---------------------------------------------------------
    W_turb_shaft = m_wf * w_turb
    W_turb_el = W_turb_shaft * spec.eta_mech * spec.eta_gen
    W_pump_shaft = m_wf * w_pump
    W_pump_el = W_pump_shaft / spec.eta_motor
    W_fan = cd["W_fan"]
    W_geo_pump_el = bp["W_el"]
    W_net = W_turb_el - W_pump_el - W_fan - W_geo_pump_el

    Q_in, Q_rec, Q_cond = hi["Q_total"], m_wf * rec["q"], cd["Q"]

    # ---- performance metrics ------------------------------------------
    T0 = spec.T_air_in_C + 273.15
    P_w0 = spec.P_geo_bar * 1e5
    geo = spec.geo_fluid
    h0 = PropsSI("H", "T", T0, "P", P_w0, geo)
    s0 = PropsSI("S", "T", T0, "P", P_w0, geo)
    T_w_in = spec.T_geo_in_C + 273.15
    s_w_in = PropsSI("S", "T", T_w_in, "P", P_w0, geo)
    Ex_geo_in = spec.m_geo * ((hi["hw_in"] - h0) - T0 * (s_w_in - s0))
    Q_geo_max = spec.m_geo * (hi["hw_in"] - h0)

    def st(T, P, h, s, x=None):
        return dict(T_C=T - 273.15, P_bar=P / 1e5, h_kJkg=h / 1e3, s_kJkgK=s / 1e3, x=x)

    h_bub = hi["h_bub"]
    s_bub = PropsSI("S", "P", P_evap, "Q", 0, wf)
    s2r = PropsSI("S", "H", h2r, "P", P_evap, wf)
    s4r = PropsSI("S", "H", h4r, "P", P_cond, wf)
    states = {
        "1  condenser outlet / feed-pump inlet": st(T1, P_cond, h1, s1),
        "2  feed-pump outlet / recuperator inlet": st(T2, P_evap, h2, s2),
        "2r recuperator outlet / preheater inlet": st(rec["T2r"], P_evap, h2r, s2r),
        "2p preheater outlet / evaporator inlet (sat. liquid)": st(T_evap, P_evap, h_bub, s_bub, 0.0),
        "3  evaporator outlet / turbine inlet": st(T3, P_evap, h3, s3),
        "4  turbine outlet / recuperator hot inlet": st(T4, P_cond, h4, s4, x4),
        "4r recuperator outlet / condenser inlet": st(rec["T4r"], P_cond, h4r, s4r),
    }

    T_out_C = hi["T_geo_out"] - 273.15
    checks = dict(
        liquid_at_preheater=dict(ok=True, subcool_K=subcool_2r, min_K=dT_liq,
                                 capped=rec["capped"] and rec["cap_reason"] == caps[0][0]),
        reinjection=dict(mode=spec.geo_outlet, target_C=spec.T_geo_out_C,
                         tol_K=(None if band is None else band[1] - spec.T_geo_out_C - 273.15),
                         floor_C=(None if band is None else band[0] - 273.15),
                         ceiling_C=(None if band is None else band[1] - 273.15),
                         T_out_C=T_out_C, ok=True, deviation_K=T_out_C - spec.T_geo_out_C),
        pinch=dict(evaporator=(ev["min_dT"], pe), preheater=(pr["min_dT"], pp),
                   recuperator=(rec["min_dT"], spec.pinch_rec_K), condenser=(cd["min_dT"], pc)),
        dry_expansion=(x4 is None or not (0.0 <= x4 < 1.0)),
        turbine_outlet_quality=x4,
    )

    return dict(
        spec=spec,
        dv=DesignVars(T_evap_C, dT_sh_eff, T_cond_C),
        P_evap_bar=P_evap_bar, P_cond_bar=P_cond_bar, m_wf=m_wf,
        states=states,
        brine=dict(states=brine_states, dP_hx_bar=spec.dP_geo_hx_bar,
                   P_prod_bar=spec.P_geo_bar, P_suction_bar=bp["P_suction_bar"],
                   P_inj_bar=bp["P_inj_bar"], dP_total_bar=bp["dP_bar"], pump=bp,
                   limiting=hi["limiting"], T_mid_C=hi["T_geo_mid"] - 273.15,
                   T_inj_C=bp["T_inj"] - 273.15),
        pump=dict(w=w_pump, W_shaft=W_pump_shaft, W_el=W_pump_el, dT=pm["perf"]["dT"]),
        turbine=dict(w=w_turb, dh_isen=tb["perf"]["dh_isen"], W_shaft=W_turb_shaft,
                     W_el=W_turb_el, x_out=x4, phase_out=tb["s2"]["phase"],
                     PR=P_evap_bar / P_cond_bar),
        recuperator=dict(**rec, Q=Q_rec, A=m_wf * rec["A_per_kg"]),
        evaporator=ev,
        preheater=pr,
        condenser=cd,
        power=dict(W_turb_shaft=W_turb_shaft, W_turb_el=W_turb_el,
                   W_pump_shaft=W_pump_shaft, W_pump_el=W_pump_el,
                   W_fan_el=W_fan, W_geo_pump_el=W_geo_pump_el,
                   W_aux=W_pump_el + W_fan + W_geo_pump_el, W_net=W_net),
        perf=dict(Q_in=Q_in, Q_evap=ev["Q"], Q_pre=pr["Q"], Q_rec=Q_rec, Q_cond=Q_cond,
                  eta_th=(W_turb_shaft - W_pump_shaft) / Q_in,
                  eta_gross_el=W_turb_el / Q_in,
                  eta_net=W_net / Q_in,
                  eta_II=W_net / Ex_geo_in, Ex_geo_in=Ex_geo_in,
                  utilisation=Q_in / Q_geo_max,
                  W_net_per_kg_geo=W_net / spec.m_geo,
                  T_geo_out_C=T_out_C, T_geo_mid_C=hi["T_geo_mid"] - 273.15,
                  P_geo_out_bar=pr["P_geo_out_bar"],          # preheater outlet = pump suction
                  T_inj_C=bp["T_inj"] - 273.15, P_inj_bar=bp["P_inj_bar"],
                  back_work_ratio=(W_pump_el + W_fan + W_geo_pump_el) / W_turb_el),
        checks=checks,
    )


# =====================================================================
#                          OPTIMISATION
# =====================================================================
def _objective(x, spec: PlantSpec, sh_fixed=None) -> float:
    """-W_net [kW] with a graded penalty for infeasible points.
    x = (T_evap, dT_sh, T_cond) or, with sh_fixed given, x = (T_evap, T_cond)."""
    if sh_fixed is None:
        T_evap, dT_sh, T_cond = float(x[0]), float(x[1]), float(x[2])
    else:
        T_evap, dT_sh, T_cond = float(x[0]), float(sh_fixed), float(x[1])
    try:
        r = solve_cycle(spec, T_evap, dT_sh, T_cond)
        return -r["power"]["W_net"] / 1e3
    except Infeasible as e:
        return 1e4 + 1e3 * e.violation
    except Exception:              # CoolProp failure far outside the envelope
        return 1e5


def design_bounds(spec: PlantSpec):
    T_air, T_in = spec.T_air_in_C, spec.T_geo_in_C
    pe = spec.pinch_evap_K
    T_cond_lo, T_cond_hi = T_air + spec.pinch_cond_K + 0.5, T_in - pe - 12.0
    T_evap_lo, T_evap_hi = T_cond_lo + 5.0, T_in - pe - SAT_VAPOUR_MARGIN_K
    dT_sh_lo, dT_sh_hi = SAT_VAPOUR_MARGIN_K, 40.0
    return (T_evap_lo, T_evap_hi), (dT_sh_lo, dT_sh_hi), (T_cond_lo, T_cond_hi)


def optimise_plant(spec: PlantSpec, grid_step_K: float = 1.0, verbose: bool = True,
                   progress=None) -> dict:
    """Maximise net power over (T_evap, T_cond) and, if spec.superheat_mode ==
    "optimise", the superheat dT_sh.

    1. coarse grid over (T_evap, T_cond) at the held superheat
       (returns a W_net map for plotting),
    2. Nelder-Mead refinement from the best grid point.

    `progress(fraction, message)` is called during the search (for a GUI).
    """
    (Te_lo, Te_hi), (sh_lo, sh_hi), (Tc_lo, Tc_hi) = design_bounds(spec)
    sh_fixed = spec.superheat_for_optimiser()
    sh_grid = sh_lo if sh_fixed is None else sh_fixed
    t0 = time.time()
    spec_full = spec
    spec = replace(spec, n_seg=max(24, spec.n_seg // 2))   # coarser T-Q grid while searching

    T_evap_grid = np.arange(np.ceil(Te_lo), Te_hi, grid_step_K)
    T_cond_grid = np.arange(np.ceil(Tc_lo), Tc_hi, grid_step_K)
    W_map = np.full((len(T_cond_grid), len(T_evap_grid)), np.nan)
    best = (None, -np.inf)
    worst = (None, np.inf, "", {})          # smallest constraint violation seen
    n_eval = 0
    for i, Tc in enumerate(T_cond_grid):
        if progress is not None:
            progress(0.9 * i / max(len(T_cond_grid), 1), f"grid search: T_cond = {Tc:.0f} degC")
        for j, Te in enumerate(T_evap_grid):
            if Te - Tc < 6.0:
                continue
            n_eval += 1
            try:
                r = solve_cycle(spec, Te, sh_grid, Tc)
                W = r["power"]["W_net"] / 1e3
                W_map[i, j] = W
                if W > best[1]:
                    best = ((Te, sh_grid, Tc), W)
            except Infeasible as e:
                if e.violation < worst[1]:
                    worst = ((Te, sh_grid, Tc), e.violation, str(e), e.info)
            except Exception:
                pass
    if verbose:
        if best[0]:
            print(f"  grid: {n_eval} evaluations in {time.time()-t0:.1f} s; best {best[1]:.1f} kW "
                  f"at T_evap={best[0][0]:.1f}, T_cond={best[0][2]:.1f} degC")
        else:
            print(f"  grid: {n_eval} evaluations, NO feasible point "
                  f"(smallest violation {worst[1]:.2f} K: {worst[2]})")

    if best[0] is None:
        return dict(feasible=False, W_map=W_map, T_evap_grid=T_evap_grid,
                    T_cond_grid=T_cond_grid, worst_violation=worst, n_eval=n_eval,
                    seconds=time.time() - t0, superheat_held=sh_fixed)

    # --- local refinement -------------------------------------------------
    if progress is not None:
        progress(0.9, "refining (Nelder-Mead)")
    Te0, sh0, Tc0 = best[0]
    if sh_fixed is None:
        x0 = np.array([Te0, sh0, Tc0])
        bounds = [(Te_lo, Te_hi), (sh_lo, sh_hi), (Tc_lo, Tc_hi)]
        simplex = np.array([x0, x0 + [1.5, 0, 0], x0 + [0, 3.0, 0], x0 + [0, 0, 1.0]])
    else:
        x0 = np.array([Te0, Tc0])
        bounds = [(Te_lo, Te_hi), (Tc_lo, Tc_hi)]
        simplex = np.array([x0, x0 + [1.5, 0], x0 + [0, 1.0]])
    simplex = np.clip(simplex, [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(_objective, x0, args=(spec, sh_fixed), method="Nelder-Mead", bounds=bounds,
                   options=dict(initial_simplex=simplex, xatol=0.02, fatol=0.02, maxfev=600))
    x_ref = res.x if -res.fun > best[1] else x0
    if sh_fixed is None:
        x_opt = (float(x_ref[0]), float(x_ref[1]), float(x_ref[2]))
    else:
        x_opt = (float(x_ref[0]), float(sh_fixed), float(x_ref[1]))
    r_opt = solve_cycle(spec_full, *x_opt)                # final design at full resolution
    if verbose:
        print(f"  refine: {res.nfev} evaluations -> W_net = {r_opt['power']['W_net']/1e3:.1f} kW "
              f"(T_evap={x_opt[0]:.2f}, dT_sh={x_opt[1]:.2f}, T_cond={x_opt[2]:.2f} degC), "
              f"total {time.time()-t0:.1f} s")
    if progress is not None:
        progress(1.0, "done")
    return dict(feasible=True, result=r_opt, x_opt=x_opt, W_map=W_map,
                T_evap_grid=T_evap_grid, T_cond_grid=T_cond_grid,
                n_eval=n_eval + res.nfev, seconds=time.time() - t0, superheat_held=sh_fixed)


# =====================================================================
#            VERIFICATION AGAINST THE RATING-MODE COMPONENT MODELS
# =====================================================================
def verify_with_rating_models(r: dict) -> list[dict]:
    """Re-rate each exchanger at its design area with the folder's rating
    functions (hex_rate / air_cooled_condenser) and compare the duty.
    Confirms the design is consistent with the digital-twin rating models."""
    spec: PlantSpec = r["spec"]
    wf, m = spec.wf, r["m_wf"]
    S = r["states"]
    h = {lbl: S[key]["h_kJkg"] * 1e3 for lbl, key in zip(STATE_LABELS, S)}
    out = []

    ev = r["evaporator"]
    rr = hex_rate(hot_fluid=spec.geo_fluid, P_h_bar=ev["P_geo_in_bar"], m_h=spec.m_geo,
                  T_h_in_C=spec.T_geo_in_C,
                  cold_fluid=wf, P_c_bar=r["P_evap_bar"], m_c=m, h_c_in_Jkg=h["2p"],
                  A_m2=ev["A"], U_W_m2K=spec.U_evap, pinch_K=spec.pinch_evap_K,
                  n_seg=spec.n_seg, name="Evaporator")
    out.append(dict(name="Evaporator (HEX.hex_rate)", Q_design=ev["Q"], Q_rating=rr["Q"],
                    extra=f"brine out {rr['T_h_out_C']:.2f} degC, limiting: {rr['limiting']}, "
                          f"pinch {rr['pinch_K_achieved']:.2f} K"))

    pr = r["preheater"]
    b1 = r["brine"]["states"][1]
    rr = hex_rate(hot_fluid=spec.geo_fluid, P_h_bar=pr["P_geo_in_bar"], m_h=spec.m_geo,
                  h_h_in_Jkg=b1["h_kJkg"] * 1e3,
                  cold_fluid=wf, P_c_bar=r["P_evap_bar"], m_c=m, h_c_in_Jkg=h["2r"],
                  A_m2=pr["A"], U_W_m2K=spec.U_pre, pinch_K=spec.pinch_pre_K,
                  n_seg=spec.n_seg, name="Preheater")
    out.append(dict(name="Preheater (HEX.hex_rate)", Q_design=pr["Q"], Q_rating=rr["Q"],
                    extra=f"brine out {rr['T_h_out_C']:.2f} degC, wf out {rr['T_c_out_C']:.2f} degC, "
                          f"limiting: {rr['limiting']}, pinch {rr['pinch_K_achieved']:.2f} K"))

    rc = r["recuperator"]
    if rc["active"]:
        rr = hex_rate(hot_fluid=wf, P_h_bar=r["P_cond_bar"], m_h=m, h_h_in_Jkg=h["4"],
                      cold_fluid=wf, P_c_bar=r["P_evap_bar"], m_c=m, h_c_in_Jkg=h["2"],
                      A_m2=rc["A"], U_W_m2K=spec.U_rec, pinch_K=spec.pinch_rec_K,
                      n_seg=spec.n_seg, name="Recuperator")
        out.append(dict(name="Recuperator (HEX.hex_rate)", Q_design=rc["Q"], Q_rating=rr["Q"],
                        extra=f"T_2r {rr['T_c_out_C']:.2f} degC, limiting: {rr['limiting']}, "
                              f"pinch {rr['pinch_K_achieved']:.2f} K"))

    cd = r["condenser"]
    if cd["T4r"] - cd["T_cond"] < 0.05:
        out.append(dict(name="Condenser (AirCooledCondenser)", Q_design=cd["Q"],
                        Q_rating=float("nan"),
                        extra="rating skipped: inlet at/inside the dome (rating model needs a (T,P) vapour inlet)"))
        return out
    try:
        ra = air_cooled_condenser(
            fluid=wf, T_in_C=cd["T4r"] - 273.15, P_in_bar=r["P_cond_bar"], m_dot=m,
            T_air_in_C=spec.T_air_in_C, m_dot_air=cd["m_air"], P_atm_bar=spec.P_atm_bar,
            A_m2=cd["A"], U_clean_W_m2K=spec.U_acc, R_foul_m2K_W=spec.R_foul_acc,
            dT_min_pinch_K=spec.pinch_cond_K, dP_air_Pa=spec.dP_air_Pa, eta_fan=spec.eta_fan)
        out.append(dict(name="Condenser (AirCooledCondenser)", Q_design=cd["Q"],
                        Q_rating=ra["hx"]["Q"],
                        extra=f"outlet: {ra['s_out']['regime']}, pinch {ra['hx']['pinch_K']:.2f} K, "
                              f"fan {ra['fan']['W_fan_elec']/1e3:.1f} kW, status {ra['status']}"))
    except ValueError as e:
        out.append(dict(name="Condenser (AirCooledCondenser)", Q_design=cd["Q"],
                        Q_rating=float("nan"), extra=f"rating failed: {e}"))
    return out


# =====================================================================
#                      DIAGRAM DATA (shared by CLI / GUI)
# =====================================================================
def ts_diagram_data(r: dict, n: int = 120) -> dict:
    """Saturation dome, the closed cycle path and the labelled state points
    in (s [kJ/kgK], T [degC]) - used by the matplotlib and Plotly plots."""
    spec: PlantSpec = r["spec"]
    wf = spec.wf
    S = r["states"]
    keys = list(S)
    P_e, P_c = r["P_evap_bar"] * 1e5, r["P_cond_bar"] * 1e5
    h = {lbl: S[key]["h_kJkg"] * 1e3 for lbl, key in zip(STATE_LABELS, keys)}
    s = {lbl: S[key]["s_kJkgK"] for lbl, key in zip(STATE_LABELS, keys)}
    T = {lbl: S[key]["T_C"] for lbl, key in zip(STATE_LABELS, keys)}

    T_tr, T_cr = PropsSI("Ttriple", wf), PropsSI("Tcrit", wf)
    Ts = np.linspace(max(T_tr + 1.0, spec.T_air_in_C + 273.15 - 40.0), T_cr - 0.3, 150)
    s_liq = PropsSI("S", "T", Ts, "Q", 0, wf) / 1e3
    s_vap = PropsSI("S", "T", Ts, "Q", 1, wf) / 1e3

    def isobar(P, h_a, h_b):
        hh = np.linspace(h_a, h_b, n)
        return PropsSI("S", "H", hh, "P", P, wf) / 1e3, PropsSI("T", "H", hh, "P", P, wf) - 273.15

    # closed path 1 -> 2 -> 2r -> 2p -> 3 -> 4 -> 4r -> 1
    s_hi, T_hi = isobar(P_e, h["2"], h["3"])      # 2 -> 2r -> 2p -> 3
    s_lo, T_lo = isobar(P_c, h["4"], h["1"])      # 4 -> 4r -> 1
    path_s = np.concatenate([[s["1"]], s_hi, [s["4"]], s_lo, [s["1"]]])
    path_T = np.concatenate([[T["1"]], T_hi, [T["4"]], T_lo, [T["1"]]])
    return dict(dome_T=Ts - 273.15, dome_s_liq=s_liq, dome_s_vap=s_vap,
                path_s=path_s, path_T=path_T,
                points=[dict(label=lbl, s=s[lbl], T=T[lbl]) for lbl in STATE_LABELS],
                T_geo_in_C=spec.T_geo_in_C, T_air_in_C=spec.T_air_in_C,
                T_crit_C=T_cr - 273.15)


def condenser_tq_data(r: dict) -> dict:
    """Piece-wise T-Q of the air-cooled condenser (from the condensate end)."""
    cd = r["condenser"]
    Q = np.array([0.0, cd["Q_sub"], cd["Q_sub"] + cd["Q_cnd"], cd["Q"]])
    T_org = np.array([cd["T1"], cd["T_cond"], cd["T_cond"], cd["T4r"]]) - 273.15
    T_air = np.array([cd["T_a_in"], cd["T_a_b"], cd["T_a_c"], cd["T_a_out"]]) - 273.15
    return dict(Q=Q, T_hot=T_org, T_cold=T_air)


def heat_train_data(r: dict) -> dict:
    """Brine-side composite T-Q: preheater (from injection end) then evaporator,
    on one cumulative-heat axis. Returns arrays in kW / degC and the split point."""
    pr, ev = r["preheater"], r["evaporator"]
    Q = np.concatenate([pr["tq_Q"], pr["Q"] + ev["tq_Q"]]) / 1e3
    T_h = np.concatenate([pr["tq_T_h"], ev["tq_T_h"]]) - 273.15
    T_c = np.concatenate([pr["tq_T_c"], ev["tq_T_c"]]) - 273.15
    return dict(Q=Q, T_hot=T_h, T_cold=T_c, Q_split_kW=pr["Q"] / 1e3)


# =====================================================================
#                             REPORTING
# =====================================================================
def print_report(r: dict, verify: bool = True) -> None:
    spec: PlantSpec = r["spec"]
    dv: DesignVars = r["dv"]
    P, pf, ck, br = r["power"], r["perf"], r["checks"], r["brine"]
    ev, pr, rc, cd = r["evaporator"], r["preheater"], r["recuperator"], r["condenser"]
    line = "=" * 80
    thin = "-" * 80
    band = spec.outlet_band_K()
    print(line)
    print(f" RECUPERATED ORC PLANT (design mode) - {spec.wf} - geothermal heat source")
    print(line)
    print(" Boundary conditions")
    print(f"   Brine ({spec.geo_fluid})   : {spec.T_geo_in_C:.1f} degC, {spec.P_geo_bar:.2f} bar, "
          f"{spec.m_geo:.1f} kg/s  -> evaporator -> {pf['T_geo_mid_C']:.2f} degC -> preheater -> "
          f"injection {pf['T_geo_out_C']:.2f} degC at {pf['P_geo_out_bar']:.2f} bar")
    if band is None:
        print("   Reinjection rule  : free (pinch-limited)")
    else:
        print(f"   Reinjection rule  : target {spec.T_geo_out_C:.1f} degC, allowed "
              f"{band[0]-273.15:.1f} .. {band[1]-273.15:.1f} degC   "
              f"(deviation {ck['reinjection']['deviation_K']:+.2f} K)")
    print(f"   Brine pressures   : production {spec.P_geo_bar:.2f} bar -> {spec.dP_geo_hx_bar:.2f} bar lost per "
          f"exchanger -> pump suction {br['P_suction_bar']:.2f} bar -> injection pump to {br['P_inj_bar']:.2f} bar "
          f"({P['W_geo_pump_el']/1e3:.1f} kW electric)")
    print(f"   Ambient air       : {spec.T_air_in_C:.1f} degC, {spec.P_atm_bar:.3f} bar")
    print(f"   Pinch             : evaporator {spec.pinch_evap_K:.1f} K, preheater {spec.pinch_pre_K:.1f} K, "
          f"recuperator {spec.pinch_rec_K:.1f} K, condenser {spec.pinch_cond_K:.1f} K    "
          f"recuperator: {'yes' if spec.recuperator else 'no'}   superheat: {spec.superheat_mode}")
    print(f"   Efficiencies      : turbine is {spec.eta_turb_is:.2f}, mech {spec.eta_mech:.2f}, "
          f"gen {spec.eta_gen:.2f} | feed pump {spec.eta_pump:.2f}, motor {spec.eta_motor:.2f} | "
          f"fan {spec.eta_fan:.2f} @ {spec.dP_air_Pa:.0f} Pa | injection pump {spec.eta_geo_pump:.2f}")
    print(thin)
    print(" Design variables")
    print(f"   T_evap = {dv.T_evap_C:7.2f} degC   (P_evap = {r['P_evap_bar']:.2f} bar)")
    print(f"   dT_sh  = {dv.dT_sh_K:7.2f} K      (turbine inlet {dv.T_evap_C + dv.dT_sh_K:.2f} degC)")
    print(f"   T_cond = {dv.T_cond_C:7.2f} degC   (P_cond = {r['P_cond_bar']:.2f} bar, "
          f"PR = {r['turbine']['PR']:.2f})")
    print(f"   m_wf   = {r['m_wf']:7.2f} kg/s   (set by {br['limiting']})")
    print(thin)
    print(" Working-fluid states")
    print(f"   {'point':52s} {'T [degC]':>9s} {'P [bar]':>8s} {'h [kJ/kg]':>10s} {'s [kJ/kgK]':>11s}")
    for name, s in r["states"].items():
        xs = f"  x={s['x']:.3f}" if (s["x"] is not None and 0 <= s["x"] <= 1) else ""
        print(f"   {name:52s} {s['T_C']:9.2f} {s['P_bar']:8.2f} {s['h_kJkg']:10.2f} "
              f"{s['s_kJkgK']:11.4f}{xs}")
    lq = ck["liquid_at_preheater"]
    print(f"   Preheater inlet (2r): liquid, subcooled {lq['subcool_K']:.2f} K "
          f"(minimum {lq['min_K']:.2f} K){' - recuperator capped to keep it liquid' if lq['capped'] else ''}")
    print(" Brine states")
    for b in br["states"]:
        print(f"   {b['point']:52s} {b['T_C']:9.2f} {b['P_bar']:8.2f} {b['h_kJkg']:10.2f}")
    print(thin)
    print(" Heat exchangers (design: pinch-limited, counter-flow)")
    print(f"   {'':14s} {'Q [kW]':>9s} {'min dT [K]':>11s} {'A [m2]':>9s} {'UA [kW/K]':>10s}   notes")
    print(f"   {'Evaporator':14s} {ev['Q']/1e3:9.1f} {ev['min_dT']:11.2f} {ev['A']:9.1f} "
          f"{spec.U_evap*ev['A']/1e3:10.1f}   boil {ev['Q_boil']/1e3:.0f} / superheat {ev['Q_sh']/1e3:.0f} kW; "
          f"pinch at {ev['pinch_location']}")
    print(f"   {'Preheater':14s} {pr['Q']/1e3:9.1f} {pr['min_dT']:11.2f} {pr['A']:9.1f} "
          f"{spec.U_pre*pr['A']/1e3:10.1f}   wf {r['states'][list(r['states'])[2]]['T_C']:.1f} -> "
          f"{dv.T_evap_C:.1f} degC (bubble point); pinch at {pr['pinch_location']}")
    if rc["active"]:
        print(f"   {'Recuperator':14s} {rc['Q']/1e3:9.1f} {rc['min_dT']:11.2f} {rc['A']:9.1f} "
              f"{spec.U_rec*rc['A']/1e3:10.1f}   effectiveness {rc['effectiveness']:.3f}"
              f"{'; duty capped by ' + rc['cap_reason'] if rc['capped'] else ''}")
    else:
        why = ("not installed" if not spec.recuperator else
               ("capped to zero by " + rc["cap_reason"]) if rc["capped"] else
               "no driving force (T_4 <= T_2 + pinch)")
        print(f"   {'Recuperator':14s} {'-':>9s} {'-':>11s} {'-':>9s} {'-':>10s}   {why}")
    print(f"   {'Condenser':14s} {cd['Q']/1e3:9.1f} {cd['min_dT']:11.2f} {cd['A']:9.1f} "
          f"{cd['U_eff']*cd['A']/1e3:10.1f}   desup {cd['Q_dsh']/1e3:.0f} / cond {cd['Q_cnd']/1e3:.0f} "
          f"/ subcool {cd['Q_sub']/1e3:.0f} kW")
    print(f"   Condenser air : {cd['m_air']:.0f} kg/s ({cd['V_air']:.0f} m3/s), "
          f"{spec.T_air_in_C:.1f} -> {cd['T_a_out']-273.15:.1f} degC, ITD = {cd['ITD']:.1f} K")
    print(thin)
    print(" Power balance")
    print(f"   Turbine  shaft / electric : {P['W_turb_shaft']/1e3:9.1f} / {P['W_turb_el']/1e3:9.1f} kW")
    print(f"   Feed pump shaft / electric: {P['W_pump_shaft']/1e3:9.1f} / {P['W_pump_el']/1e3:9.1f} kW")
    print(f"   ACC fans electric         : {'':11s} {P['W_fan_el']/1e3:9.1f} kW")
    print(f"   Injection pump electric   : {'':11s} {P['W_geo_pump_el']/1e3:9.1f} kW   "
          f"(head {br['dP_total_bar']:.2f} bar: {br['P_suction_bar']:.2f} -> {br['P_inj_bar']:.2f} bar)")
    print(f"   NET ELECTRIC POWER        : {'':11s} {P['W_net']/1e3:9.1f} kW")
    print(thin)
    print(" Performance")
    print(f"   Heat from brine           : {pf['Q_in']/1e3:9.1f} kW   (evaporator {pf['Q_evap']/1e3:.0f} + "
          f"preheater {pf['Q_pre']/1e3:.0f}; utilisation {100*pf['utilisation']:.1f} % of heat above "
          f"{spec.T_air_in_C:.0f} degC)")
    print(f"   Recuperated heat          : {pf['Q_rec']/1e3:9.1f} kW")
    print(f"   Heat rejected             : {pf['Q_cond']/1e3:9.1f} kW")
    print(f"   Cycle thermal eff.        : {100*pf['eta_th']:9.2f} %   (shaft work / Q_in)")
    print(f"   Net electric eff.         : {100*pf['eta_net']:9.2f} %   (W_net / Q_in)")
    print(f"   Exergy (2nd-law) eff.     : {100*pf['eta_II']:9.2f} %   "
          f"(W_net / brine exergy {pf['Ex_geo_in']/1e3:.0f} kW at T0 = {spec.T_air_in_C:.0f} degC)")
    print(f"   Back-work ratio           : {100*pf['back_work_ratio']:9.2f} %   ((pumps+fans)/turbine)")
    print(f"   Specific net power        : {pf['W_net_per_kg_geo']/1e3:9.2f} kW per kg/s brine")
    print(f"   Brine to injection well   : {pf['T_inj_C']:9.2f} degC at {pf['P_inj_bar']:.2f} bar "
          f"(preheater outlet {pf['T_geo_out_C']:.2f} degC at {pf['P_geo_out_bar']:.2f} bar)")
    if verify:
        print(thin)
        print(" Verification: re-rating each exchanger at its design area with the rating models")
        for v in verify_with_rating_models(r):
            if np.isnan(v["Q_rating"]):
                print(f"   {v['name']:32s} Q design {v['Q_design']/1e3:8.1f} kW | {v['extra']}")
                continue
            dev = 100 * (v["Q_rating"] - v["Q_design"]) / v["Q_design"]
            print(f"   {v['name']:32s} Q design {v['Q_design']/1e3:8.1f} kW | rating "
                  f"{v['Q_rating']/1e3:8.1f} kW ({dev:+.2f} %)  {v['extra']}")
    print(line)


def summary_row(r: dict) -> dict:
    spec, dv, P, pf = r["spec"], r["dv"], r["power"], r["perf"]
    return dict(mode=spec.geo_outlet, recuperator=spec.recuperator, superheat_mode=spec.superheat_mode,
                T_evap_C=dv.T_evap_C, dT_sh_K=dv.dT_sh_K, T_cond_C=dv.T_cond_C,
                P_evap_bar=r["P_evap_bar"], P_cond_bar=r["P_cond_bar"], m_wf=r["m_wf"],
                W_turb_el_kW=P["W_turb_el"] / 1e3, W_pump_el_kW=P["W_pump_el"] / 1e3,
                W_fan_kW=P["W_fan_el"] / 1e3, W_geo_pump_kW=P["W_geo_pump_el"] / 1e3,
                W_net_kW=P["W_net"] / 1e3,
                Q_in_kW=pf["Q_in"] / 1e3, Q_evap_kW=pf["Q_evap"] / 1e3, Q_pre_kW=pf["Q_pre"] / 1e3,
                Q_rec_kW=pf["Q_rec"] / 1e3, Q_cond_kW=pf["Q_cond"] / 1e3,
                eta_th=pf["eta_th"], eta_net=pf["eta_net"], eta_II=pf["eta_II"],
                T_geo_mid_C=pf["T_geo_mid_C"], T_geo_out_C=pf["T_geo_out_C"],
                P_geo_out_bar=pf["P_geo_out_bar"], T_inj_C=pf["T_inj_C"], P_inj_bar=pf["P_inj_bar"],
                pump_head_bar=r["brine"]["dP_total_bar"],
                subcool_preheater_inlet_K=r["checks"]["liquid_at_preheater"]["subcool_K"],
                m_air=r["condenser"]["m_air"], A_evap_m2=r["evaporator"]["A"],
                A_pre_m2=r["preheater"]["A"], A_rec_m2=r["recuperator"]["A"],
                A_acc_m2=r["condenser"]["A"])


def result_to_dict(r: dict) -> dict:
    """JSON-serialisable summary of a design (profiles excluded)."""
    def strip(d):
        return {k: v for k, v in d.items() if not k.startswith("tq_")}
    return dict(spec=asdict(r["spec"]), design=asdict(r["dv"]), summary=summary_row(r),
                states=r["states"], brine=r["brine"], checks=r["checks"],
                evaporator=strip(r["evaporator"]), preheater=strip(r["preheater"]),
                recuperator=strip(r["recuperator"]), condenser=r["condenser"],
                pump=r["pump"], turbine=r["turbine"], power=r["power"], perf=r["perf"])


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)


def export_json(r: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(result_to_dict(r), f, indent=2, default=_json_default)


# =====================================================================
#                          PLOTS (matplotlib, CLI)
# =====================================================================
def plot_cycle(r: dict, filename: str = "ORC_recuperated_cycle.png", show: bool = False):
    import matplotlib.pyplot as plt

    spec: PlantSpec = r["spec"]
    wf = spec.wf
    fig, axs = plt.subplots(2, 3, figsize=(19, 10))
    HOT, COLD = "#eb6834", "#2a78d6"

    # ---- (a) T-s diagram ----------------------------------------------
    d = ts_diagram_data(r)
    ax = axs[0, 0]
    ax.plot(d["dome_s_liq"], d["dome_T"], "k-", lw=1)
    ax.plot(d["dome_s_vap"], d["dome_T"], "k-", lw=1, label="saturation dome")
    ax.plot(d["path_s"], d["path_T"], "-", color=COLD, lw=2, label="cycle")
    for p in d["points"]:
        ax.plot(p["s"], p["T"], "ko", ms=4)
        ax.annotate(p["label"], (p["s"], p["T"]), textcoords="offset points", xytext=(5, 4), fontsize=9)
    ax.axhline(spec.T_geo_in_C, color=HOT, ls=":", lw=1, label="brine in")
    ax.axhline(spec.T_air_in_C, color="#1baf7a", ls=":", lw=1, label="ambient air")
    ss = [p["s"] for p in d["points"]]
    ax.set_xlim(min(ss) - 0.4, max(ss) + 0.3)
    ax.set_ylim(spec.T_air_in_C - 10, max(spec.T_geo_in_C, r["dv"].T_evap_C) + 15)
    ax.set_xlabel("s [kJ/(kg K)]"); ax.set_ylabel("T [degC]")
    ax.set_title(f"T-s diagram ({wf})  -  W_net = {r['power']['W_net']/1e3:.0f} kW")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")

    # ---- (b) brine train composite -----------------------------------
    t = heat_train_data(r)
    ax = axs[0, 1]
    ax.plot(t["Q"], t["T_hot"], color=HOT, lw=2, label=f"brine ({spec.geo_fluid})")
    ax.plot(t["Q"], t["T_cold"], color=COLD, lw=2, label=wf)
    ax.axvline(t["Q_split_kW"], color="gray", ls="--", lw=1)
    ax.text(t["Q_split_kW"] / 2, ax.get_ylim()[1] - 3, "preheater", ha="center", fontsize=9, color="gray")
    ax.text((t["Q_split_kW"] + t["Q"][-1]) / 2, ax.get_ylim()[1] - 3, "evaporator", ha="center", fontsize=9, color="gray")
    ax.set_title(f"Brine train T-Q   preheater {r['preheater']['Q']/1e3:.0f} + evaporator {r['evaporator']['Q']/1e3:.0f} kW")
    ax.set_xlabel("Q [kW]  (from injection end)"); ax.set_ylabel("T [degC]")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ---- (c) evaporator T-Q -------------------------------------------
    ev = r["evaporator"]
    ax = axs[0, 2]
    ax.plot(ev["tq_Q"] / 1e3, ev["tq_T_h"] - 273.15, color=HOT, lw=2, label="brine")
    ax.plot(ev["tq_Q"] / 1e3, ev["tq_T_c"] - 273.15, color=COLD, lw=2, label=wf)
    ax.set_title(f"Evaporator T-Q   Q = {ev['Q']/1e3:.0f} kW, min dT = {ev['min_dT']:.1f} K")
    ax.set_xlabel("Q [kW]  (from brine outlet end)"); ax.set_ylabel("T [degC]")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ---- (d) preheater T-Q --------------------------------------------
    pr = r["preheater"]
    ax = axs[1, 0]
    ax.plot(pr["tq_Q"] / 1e3, pr["tq_T_h"] - 273.15, color=HOT, lw=2, label="brine")
    ax.plot(pr["tq_Q"] / 1e3, pr["tq_T_c"] - 273.15, color=COLD, lw=2, label=wf)
    ax.set_title(f"Preheater T-Q   Q = {pr['Q']/1e3:.0f} kW, min dT = {pr['min_dT']:.1f} K")
    ax.set_xlabel("Q [kW]  (from injection end)"); ax.set_ylabel("T [degC]")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ---- (e) recuperator T-Q ------------------------------------------
    rc = r["recuperator"]
    ax = axs[1, 1]
    if rc["active"]:
        Q = rc["tq_Q"] * r["m_wf"] / 1e3
        ax.plot(Q, rc["tq_T_h"] - 273.15, color=HOT, lw=2, label="turbine exhaust (4 -> 4r)")
        ax.plot(Q, rc["tq_T_c"] - 273.15, color=COLD, lw=2, label="pump discharge (2 -> 2r)")
        ax.set_title(f"Recuperator T-Q   Q = {rc['Q']/1e3:.0f} kW, min dT = {rc['min_dT']:.1f} K, "
                     f"eff. = {rc['effectiveness']:.2f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "recuperator inactive", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Recuperator T-Q")
    ax.set_xlabel("Q [kW]  (from cold end)"); ax.set_ylabel("T [degC]"); ax.grid(alpha=0.3)

    # ---- (f) condenser T-Q --------------------------------------------
    cd = r["condenser"]
    c = condenser_tq_data(r)
    ax = axs[1, 2]
    ax.plot(c["Q"] / 1e3, c["T_hot"], "-o", color=HOT, lw=2, ms=4, label=f"{wf} (4r -> 1)")
    ax.plot(c["Q"] / 1e3, c["T_cold"], "-s", color=COLD, lw=2, ms=4, label=f"air, {cd['m_air']:.0f} kg/s")
    ax.set_title(f"Air-cooled condenser T-Q   Q = {cd['Q']/1e3:.0f} kW, min dT = {cd['min_dT']:.1f} K, "
                 f"fans {cd['W_fan']/1e3:.0f} kW")
    ax.set_xlabel("Q [kW]  (from condensate / air-inlet end)"); ax.set_ylabel("T [degC]")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle(f"Recuperated ORC on geothermal brine {spec.T_geo_in_C:.0f} degC, {spec.m_geo:.0f} kg/s  |  "
                 f"reinjection {r['perf']['T_geo_out_C']:.1f} degC ({spec.geo_outlet})  |  "
                 f"ambient {spec.T_air_in_C:.0f} degC", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return filename


def plot_map(opt: dict, spec: PlantSpec, filename: str = "ORC_recuperated_map.png", show: bool = False):
    import matplotlib.pyplot as plt

    W = opt["W_map"]
    if np.all(np.isnan(W)):
        return None
    floor = -100.0
    Wp = np.clip(W, floor, None)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    X, Y = np.meshgrid(opt["T_evap_grid"], opt["T_cond_grid"])
    cf = ax.contourf(X, Y, Wp, levels=25, cmap="Blues")
    fig.colorbar(cf, ax=ax, label=f"Net electric power [kW]  (clipped below {floor:.0f} kW)")
    cs = ax.contour(X, Y, Wp, levels=10, colors="w", linewidths=0.6)
    ax.clabel(cs, fmt="%.0f", fontsize=7)
    ax.contour(X, Y, W, levels=[0.0], colors="#52514e", linewidths=1.2, linestyles="--")
    if opt.get("feasible"):
        x = opt["x_opt"]
        ax.plot(x[0], x[2], "*", color="#eb6834", ms=14,
                label=f"optimum {opt['result']['power']['W_net']/1e3:.0f} kW")
        ax.plot([], [], "--", color="#52514e", label="W_net = 0")
        ax.legend(loc="lower right")
    ax.set_xlabel("Evaporation temperature T_evap [degC]")
    ax.set_ylabel("Condensing temperature T_cond [degC]")
    ax.set_title(f"Net power map  |  brine {spec.T_geo_in_C:.0f} degC, outlet {spec.geo_outlet}, "
                 f"ambient {spec.T_air_in_C:.0f} degC", fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return filename


# =====================================================================
#                                CLI
# =====================================================================
def _parse():
    p = argparse.ArgumentParser(description="Recuperated ORC plant model (design mode) for an geothermal heat source")
    p.add_argument("--geo-outlet", choices=["band", "free", "fixed"], default="band",
                   help="'band': reinjection target +/- tolerance; 'free': pinch only; 'fixed': band with 0 tolerance")
    p.add_argument("--T-geo-in", type=float, default=100.0, help="brine production temperature [degC]")
    p.add_argument("--T-geo-out", type=float, default=60.0, help="reinjection target [degC]")
    p.add_argument("--T-geo-out-tol", type=float, default=10.0, help="allowed deviation of the reinjection T [K]")
    p.add_argument("--P-geo", type=float, default=5.0, help="production wellhead pressure [bar]")
    p.add_argument("--P-inj", type=float, default=5.0, help="injection pump discharge pressure [bar]")
    p.add_argument("--dP-geo-hx", type=float, default=0.3, help="brine pressure drop per exchanger [bar]")
    p.add_argument("--m-geo", type=float, default=50.0, help="brine mass flow [kg/s]")
    p.add_argument("--fluid", default="Isobutane", help="working fluid (CoolProp name)")
    p.add_argument("--superheat", choices=list(SUPERHEAT_MODES), default="none")
    p.add_argument("--dT-sh-fixed", type=float, default=0.0, help="superheat when --superheat fixed [K]")
    p.add_argument("--pinch", type=float, default=10.0, help="minimum approach in every HX [K]")
    p.add_argument("--pinch-evap", type=float); p.add_argument("--pinch-pre", type=float)
    p.add_argument("--pinch-rec", type=float); p.add_argument("--pinch-cond", type=float)
    p.add_argument("--dT-liquid-min", type=float, default=2.0, help="min subcooling at preheater inlet [K]")
    p.add_argument("--ambient", type=float, default=25.0, help="ambient air temperature [degC]")
    p.add_argument("--no-recuperator", action="store_true")
    p.add_argument("--T-evap", type=float, help="evaluate a fixed design point instead of optimising")
    p.add_argument("--dT-sh", type=float, default=0.0)
    p.add_argument("--T-cond", type=float)
    p.add_argument("--grid-step", type=float, default=1.0, help="grid resolution for the optimiser [K]")
    p.add_argument("--compare", action="store_true", help="4-way table: fixed/free x recuperated/simple")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--show", action="store_true", help="open the figures interactively")
    p.add_argument("--json", help="write the design summary to this file")
    return p.parse_args()


def main():
    a = _parse()
    spec = PlantSpec(T_geo_in_C=a.T_geo_in, T_geo_out_C=a.T_geo_out, T_geo_out_tol_K=a.T_geo_out_tol,
                     P_geo_bar=a.P_geo, P_inj_bar=a.P_inj, dP_geo_hx_bar=a.dP_geo_hx, m_geo=a.m_geo,
                     geo_outlet=a.geo_outlet,
                     wf=a.fluid, superheat_mode=a.superheat, dT_sh_fixed_K=a.dT_sh_fixed,
                     pinch_evap_K=a.pinch_evap or a.pinch, pinch_pre_K=a.pinch_pre or a.pinch,
                     pinch_rec_K=a.pinch_rec or a.pinch, pinch_cond_K=a.pinch_cond or a.pinch,
                     dT_liquid_min_K=a.dT_liquid_min, T_air_in_C=a.ambient,
                     recuperator=not a.no_recuperator)

    if a.compare:
        rows = []
        for mode in ("fixed", "free"):
            for rec in (True, False):
                sp = replace(spec, geo_outlet=mode, recuperator=rec)
                print(f"\n>>> optimising: brine outlet {mode}, recuperator {'yes' if rec else 'no'}")
                opt = optimise_plant(sp, grid_step_K=a.grid_step)
                if opt["feasible"]:
                    rows.append(summary_row(opt["result"]))
                else:
                    rows.append(dict(mode=mode, recuperator=rec, infeasible=True,
                                     violation=opt["worst_violation"][1]))
        print("\n" + "=" * 126)
        print(" SUMMARY (pinch evap/pre/rec/cond = %.0f/%.0f/%.0f/%.0f K, ambient %.0f degC, brine %.0f degC / %.0f kg/s)"
              % (spec.pinch_evap_K, spec.pinch_pre_K, spec.pinch_rec_K, spec.pinch_cond_K,
                 spec.T_air_in_C, spec.T_geo_in_C, spec.m_geo))
        print("=" * 126)
        print(f" {'outlet':6s} {'recup':5s} {'T_evap':>7s} {'dT_sh':>6s} {'T_cond':>7s} {'m_wf':>6s} "
              f"{'W_turb':>7s} {'W_pump':>7s} {'W_fan':>6s} {'W_brine':>7s} {'W_NET':>7s} {'Q_in':>7s} "
              f"{'Q_rec':>6s} {'eta_th':>6s} {'eta_net':>7s} {'T_inj':>6s}")
        print(f" {'':6s} {'':5s} {'degC':>7s} {'K':>6s} {'degC':>7s} {'kg/s':>6s} {'kW':>7s} {'kW':>7s} "
              f"{'kW':>6s} {'kW':>7s} {'kW':>7s} {'kW':>7s} {'kW':>6s} {'%':>6s} {'%':>7s} {'degC':>6s}")
        for w in rows:
            if w.get("infeasible"):
                print(f" {w['mode']:6s} {'yes' if w['recuperator'] else 'no':5s}   INFEASIBLE "
                      f"(smallest constraint violation {w['violation']:.2f} K)")
                continue
            print(f" {w['mode']:6s} {'yes' if w['recuperator'] else 'no':5s} {w['T_evap_C']:7.1f} "
                  f"{w['dT_sh_K']:6.1f} {w['T_cond_C']:7.1f} {w['m_wf']:6.1f} {w['W_turb_el_kW']:7.1f} "
                  f"{w['W_pump_el_kW']:7.1f} {w['W_fan_kW']:6.1f} {w['W_geo_pump_kW']:7.1f} {w['W_net_kW']:7.1f} "
                  f"{w['Q_in_kW']:7.0f} {w['Q_rec_kW']:6.0f} {100*w['eta_th']:6.2f} "
                  f"{100*w['eta_net']:7.2f} {w['T_geo_out_C']:6.1f}")
        print("=" * 126)
        return

    if a.T_evap is not None and a.T_cond is not None:
        r = solve_cycle(spec, a.T_evap, a.dT_sh, a.T_cond)
        opt = None
    else:
        print(f">>> optimising net power: reinjection rule {spec.geo_outlet}, "
              f"recuperator {'yes' if spec.recuperator else 'no'}, superheat {spec.superheat_mode}, "
              f"ambient {spec.T_air_in_C:.0f} degC")
        opt = optimise_plant(spec, grid_step_K=a.grid_step)
        if not opt["feasible"]:
            x, v, msg, info = opt["worst_violation"]
            print(f"\nNo feasible design: the closest point (T_evap={x[0]:.1f}, T_cond={x[2]:.1f} degC) "
                  f"still violates a constraint by {v:.2f} K: {msg}")
            sys.exit(1)
        r = opt["result"]

    print()
    print_report(r)
    if a.json:
        export_json(r, a.json)
        print(f"summary written to {a.json}")
    if not a.no_plot:
        tag = f"{spec.geo_outlet}{'' if spec.recuperator else '_norecup'}"
        f1 = plot_cycle(r, f"ORC_recuperated_cycle_{tag}.png", show=a.show)
        print(f"figure written to {f1}")
        if opt is not None:
            f2 = plot_map(opt, spec, f"ORC_recuperated_map_{tag}.png", show=a.show)
            if f2:
                print(f"figure written to {f2}")


if __name__ == "__main__":
    main()
