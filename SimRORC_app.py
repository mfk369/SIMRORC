"""
SimRORC - Simulated Recuperated ORC (public web version).

Design-mode simulator for a recuperated organic Rankine cycle on a liquid
geothermal or waste-heat source. You give the boundary conditions (brine, ambient,
working fluid, pinch temperatures, efficiencies, reinjection target +/- tolerance)
and the tool sizes the plant for maximum net power, guaranteeing:
  * brine train = EVAPORATOR then PREHEATER (pressure drop in each, brine pump restores it),
  * the preheater brings the working fluid exactly to its bubble point,
  * the working fluid is liquid before the preheater (state 2r subcooled),
  * the minimum approach (pinch) in every exchanger,
  * the reinjection temperature inside the allowed band.

Author: Md Faisal Karim - https://www.linkedin.com/in/md-faisal-karim
Run:    streamlit run SimRORC_app.py
Model:  ORC_recuperated.py (composes Turbine.py, Pump.py, HEX.py, AirCooledCondenser.py)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from CoolProp.CoolProp import PropsSI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ORC_recuperated import (PlantSpec, WORKING_FLUIDS, STATE_LABELS, solve_cycle,  # noqa: E402
                             optimise_plant, Infeasible, ts_diagram_data, condenser_tq_data,
                             heat_train_data, result_to_dict, verify_with_rating_models,
                             design_bounds, _json_default)

APP_NAME = "Simulated Recuperated ORC (SimRORC)"
AUTHOR = "Md Faisal Karim"
LINKEDIN = "https://www.linkedin.com/in/md-faisal-karim"
GRID_STEP_K = 2.0          # optimiser grid step used on the web (fast)

# ---------------------------------------------------------------------
# palette (validated data-viz reference palette)
# ---------------------------------------------------------------------
C_HOT, C_COLD, C_AQUA = "#eb6834", "#2a78d6", "#1baf7a"
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_SURF, C_PLANE = "#fcfcfb", "#f1f0ec"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
WIDE = ({"width": "stretch"} if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 49)
        else {"use_container_width": True})   # new/old Streamlit keyword for full-width elements

st.set_page_config(page_title="SimRORC", page_icon="♨️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
  div[data-testid="stMetric"] {{
      background: {C_PLANE}; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
      padding: 10px 14px; }}
  div[data-testid="stMetric"] label {{ color: {C_INK2}; font-size: 0.78rem; }}
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p {{
      white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
      line-height: 1.15; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
  div[data-testid="stMetricDelta"] {{ font-size: 0.78rem; }}
  .hero {{ font-size: 2.6rem; font-weight: 600; line-height: 1.1; color: {C_INK}; }}
  .hero-sub {{ color: {C_INK2}; font-size: 0.95rem; }}
  .small {{ color: {C_INK2}; font-size: 0.85rem; }}
  .byline {{ color: {C_INK2}; font-size: 0.95rem; margin-top: -0.4rem; }}
  .byline a {{ color: {C_COLD}; text-decoration: none; }}
  .footer {{ color: {C_INK2}; font-size: 0.8rem; border-top: 1px solid rgba(11,11,11,0.10);
             padding-top: 0.6rem; margin-top: 1.5rem; }}
  section[data-testid="stSidebar"] .stNumberInput label {{ font-size: 0.82rem; }}
</style>
""", unsafe_allow_html=True)


# =====================================================================
#                              SIDEBAR
# =====================================================================
def sidebar():
    with st.sidebar:
        st.markdown("## ♨️ SimRORC")
        st.markdown(f'<div class="small">Simulated Recuperated ORC · by <b>{AUTHOR}</b> · '
                    f'<a href="{LINKEDIN}" target="_blank">LinkedIn</a></div>', unsafe_allow_html=True)
        st.caption("Set the boundary conditions, then run. Design mode: the plant is sized "
                   "from the inputs; nothing is rated.")
        run = st.button("▶  Run design", type="primary", **WIDE)

        with st.expander("Brine (heat source)", expanded=True):
            T_geo_in = st.number_input("Production temperature [°C]", 40.0, 300.0, 100.0, 1.0)
            m_geo = st.number_input("Mass flow [kg/s]", 0.1, 5000.0, 50.0, 1.0)
            P_geo = st.number_input("Pressure held by brine pump [bar]", 1.0, 200.0, 5.0, 0.5,
                                    help="brine pressure at the evaporator inlet; must be above the boiling "
                                         "pressure of water at the production temperature")
            P_sat_geo = PropsSI("P", "T", T_geo_in + 273.15, "Q", 0, "Water") / 1e5
            if P_geo <= P_sat_geo + 0.2:
                st.warning(f"Water at {T_geo_in:.0f} °C boils at {P_sat_geo:.1f} bar. Set the brine pressure "
                           f"above that (for example {P_sat_geo + 3:.0f} bar), otherwise the brine flashes to steam.",
                           icon="⚠️")
            rule = st.radio("Reinjection rule",
                            ["Target ± tolerance", "Exact target", "Free (pinch decides)"], index=0,
                            help="Target ± tolerance: the brine leaves inside [target − tol, target + tol]. "
                                 "Exact: forced to the target. Free: only the pinch limits the cooling.")
            T_geo_out = st.number_input("Reinjection target [°C]", 0.0, 300.0, 60.0, 1.0,
                                        disabled=(rule == "Free (pinch decides)"))
            tol = st.number_input("Allowed deviation ± [K]", 0.0, 150.0, 10.0, 1.0,
                                  disabled=(rule != "Target ± tolerance"))

        with st.expander("Brine circuit losses & pump", expanded=False):
            dP_hx = st.number_input("Pressure drop per exchanger [bar]", 0.0, 5.0, 0.3, 0.05,
                                    help="lost in the evaporator and again in the preheater")
            dP_extra = st.number_input("Other loop losses [bar]", 0.0, 50.0, 0.0, 0.1,
                                       help="any extra head the brine pump must supply")
            eta_bp = st.number_input("Brine pump efficiency [-]", 0.2, 1.0, 0.75, 0.05)
            eta_bm = st.number_input("Brine pump motor [-]", 0.5, 1.0, 0.95, 0.01)

        with st.expander("Working fluid & cycle", expanded=False):
            wf = st.selectbox("Working fluid (CoolProp)", WORKING_FLUIDS, index=0)
            recup = st.toggle("Recuperator installed", value=True)
            sh_mode = st.radio("Superheat at turbine inlet",
                               ["None (saturated vapour)", "Optimise", "Fixed value"], index=0,
                               help="The evaporator only boils the fluid; any superheat happens in its last section.")
            sh_fixed = st.number_input("Fixed superheat [K]", 0.0, 100.0, 0.0, 0.5,
                                       disabled=(sh_mode != "Fixed value"))
            dT_sc = st.number_input("Condensate subcooling [K]", 0.0, 30.0, 0.0, 0.5)
            dT_liq = st.number_input("Min. subcooling at preheater inlet [K]", 0.05, 40.0, 2.0, 0.5,
                                     help="Guarantees no vapour before the preheater: the recuperator duty is "
                                          "capped so state 2r stays at least this far below the bubble point.")

        with st.expander("Pinch (minimum approach) temperatures", expanded=False):
            p_e = st.number_input("Evaporator pinch [K]", 0.5, 50.0, 10.0, 0.5)
            p_p = st.number_input("Preheater pinch [K]", 0.5, 50.0, 10.0, 0.5)
            p_r = st.number_input("Recuperator pinch [K]", 0.5, 50.0, 10.0, 0.5)
            p_c = st.number_input("Condenser pinch [K]", 0.5, 50.0, 10.0, 0.5)

        with st.expander("Ambient & air-cooled condenser", expanded=False):
            T_air = st.number_input("Ambient air temperature [°C]", -30.0, 60.0, 25.0, 1.0)
            P_atm = st.number_input("Atmospheric pressure [bar]", 0.5, 1.2, 1.01325, 0.005, format="%.5f")
            dP_air = st.number_input("Fan pressure rise [Pa]", 10.0, 1000.0, 150.0, 10.0)
            eta_fan = st.number_input("Fan efficiency [-]", 0.1, 1.0, 0.65, 0.05)
            U_acc = st.number_input("ACC overall U [W/m²K]", 10.0, 5000.0, 850.0, 10.0)
            R_foul = st.number_input("ACC fouling resistance [m²K/W]", 0.0, 0.01, 0.0, 0.0001, format="%.4f")

        with st.expander("Turbomachinery efficiencies", expanded=False):
            eta_t = st.number_input("Turbine isentropic [-]", 0.3, 1.0, 0.88, 0.01)
            eta_m = st.number_input("Turbine mechanical [-]", 0.5, 1.0, 0.98, 0.01)
            eta_g = st.number_input("Generator [-]", 0.5, 1.0, 0.97, 0.01)
            eta_p = st.number_input("Feed pump isentropic [-]", 0.2, 1.0, 0.80, 0.01)
            eta_mo = st.number_input("Feed pump motor [-]", 0.5, 1.0, 0.95, 0.01)

        with st.expander("Heat-exchanger U values (areas only)", expanded=False):
            U_evap = st.number_input("Evaporator U [W/m²K]", 50.0, 10000.0, 1000.0, 50.0)
            U_pre = st.number_input("Preheater U [W/m²K]", 50.0, 10000.0, 800.0, 50.0)
            U_rec = st.number_input("Recuperator U [W/m²K]", 20.0, 5000.0, 300.0, 10.0)
            n_seg = st.number_input("T-Q segments per exchanger", 20, 300, 60, 10)

        with st.expander("Design variables", expanded=True):
            mode = st.radio("Design point", ["Optimise for maximum net power", "Manual"], index=0,
                            help=f"Optimise: grid search ({GRID_STEP_K:.0f} K steps over evaporation and condensing "
                                 "temperature) followed by a local refinement.")
            manual = None
            if mode == "Manual":
                T_evap = st.number_input("Evaporation temperature T_evap [°C]", 0.0, 300.0, 65.0, 0.5)
                dT_sh = st.number_input("Superheat at turbine inlet [K]", 0.0, 100.0, 0.0, 0.5)
                T_cond = st.number_input("Condensing temperature T_cond [°C]", -20.0, 200.0, 43.0, 0.5)
                manual = (T_evap, dT_sh, T_cond)

    geo_outlet = {"Target ± tolerance": "band", "Exact target": "fixed", "Free (pinch decides)": "free"}[rule]
    superheat_mode = {"None (saturated vapour)": "none", "Optimise": "optimise", "Fixed value": "fixed"}[sh_mode]
    spec = PlantSpec(T_geo_in_C=T_geo_in, P_geo_bar=P_geo, m_geo=m_geo, T_geo_out_C=T_geo_out,
                     T_geo_out_tol_K=tol, geo_outlet=geo_outlet,
                     dP_geo_hx_bar=dP_hx, dP_geo_extra_bar=dP_extra, eta_geo_pump=eta_bp, eta_geo_motor=eta_bm,
                     wf=wf, recuperator=recup, dT_subcool_K=dT_sc, dT_liquid_min_K=dT_liq,
                     superheat_mode=superheat_mode, dT_sh_fixed_K=sh_fixed,
                     pinch_evap_K=p_e, pinch_pre_K=p_p, pinch_rec_K=p_r, pinch_cond_K=p_c,
                     T_air_in_C=T_air, P_atm_bar=P_atm, dP_air_Pa=dP_air, eta_fan=eta_fan,
                     U_acc=U_acc, R_foul_acc=R_foul,
                     eta_turb_is=eta_t, eta_mech=eta_m, eta_gen=eta_g, eta_pump=eta_p, eta_motor=eta_mo,
                     U_evap=U_evap, U_pre=U_pre, U_rec=U_rec, n_seg=int(n_seg))
    return spec, dict(run=run, mode=mode, manual=manual, grid=GRID_STEP_K)


# =====================================================================
#                          COMPUTATION
# =====================================================================
def _optimise_with_cache(spec: PlantSpec, grid: float) -> dict:
    """Session-level cache (identical inputs -> instant); shows a progress bar
    while a new optimisation runs (st.cache_data cannot replay the bar)."""
    key = json.dumps(asdict(spec), sort_keys=True) + f"|{grid}"
    cache = st.session_state.setdefault("opt_cache", {})
    if key in cache:
        return cache[key]
    bar = st.progress(0.0, text="starting optimisation …")

    def cb(frac, msg):
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    opt = optimise_plant(spec, grid_step_K=grid, verbose=False, progress=cb)
    bar.empty()
    if len(cache) > 20:
        cache.pop(next(iter(cache)))
    cache[key] = opt
    return opt


def compute(spec: PlantSpec, ctl: dict) -> dict:
    out = dict(spec=spec, mode=ctl["mode"], result=None, opt=None, error=None, info={},
               when=time.strftime("%Y-%m-%d %H:%M:%S"))
    P_sat_geo = PropsSI("P", "T", spec.T_geo_in_C + 273.15, "Q", 0, spec.geo_fluid) / 1e5
    if spec.P_geo_bar <= P_sat_geo + 0.2:
        out["error"] = (f"The brine would flash to steam: water at {spec.T_geo_in_C:.0f} °C boils at "
                        f"{P_sat_geo:.1f} bar, but the brine pressure is only {spec.P_geo_bar:.1f} bar. "
                        f"Raise **Pressure held by brine pump** above {P_sat_geo:.1f} bar "
                        f"(for example {P_sat_geo + 3:.0f} bar) and run again.")
        return out
    try:
        if ctl["mode"] == "Manual":
            out["result"] = solve_cycle(spec, *ctl["manual"])
        else:
            opt = _optimise_with_cache(spec, ctl["grid"])
            out["opt"] = opt
            if opt["feasible"]:
                out["result"] = opt["result"]
            else:
                x, v, msg, info = opt["worst_violation"]
                out["error"] = (f"No feasible design in the whole search space. The closest point "
                                f"(T_evap = {x[0]:.1f} °C, T_cond = {x[2]:.1f} °C) still violates a "
                                f"constraint by {v:.2f} K: {msg}")
                out["info"] = info
    except Infeasible as e:
        out["error"] = f"This design point is infeasible: {e}"
        out["info"] = e.info
    except Exception:
        out["error"] = "Unexpected error:\n\n```\n" + traceback.format_exc() + "\n```"
    return out


# =====================================================================
#                          PLANT SCHEMATIC (SVG)
# =====================================================================
def plant_svg(r: dict | None) -> str:
    """Block-flow schematic with the live state values.
    Brine: production -> brine pump -> EVAPORATOR -> PREHEATER -> injection.
    Working fluid: pump -> recuperator (cold) -> preheater -> evaporator -> turbine -> recuperator (hot) -> condenser."""
    if r is not None:
        S = r["states"]
        sp = r["spec"]; P = r["power"]
        ev = r["evaporator"]; pr = r["preheater"]; rc = r["recuperator"]; cd = r["condenser"]; br = r["brine"]
        T = {lbl: S[key]["T_C"] for lbl, key in zip(STATE_LABELS, S)}
        P_hi, P_lo = r["P_evap_bar"], r["P_cond_bar"]
        b0, b1, b2 = br["states"]
        geo_in = f"{b0['T_C']:.1f} °C · {sp.m_geo:.0f} kg/s · {b0['P_bar']:.1f} bar"
        geo_mid = f"{b1['T_C']:.1f} °C · {b1['P_bar']:.1f} bar"
        geo_out = f"{b2['T_C']:.1f} °C · {b2['P_bar']:.1f} bar"
        q_ev, q_pr = f"Q = {ev['Q']/1e3:,.0f} kW", f"Q = {pr['Q']/1e3:,.0f} kW"
        q_rc = f"Q = {rc['Q']/1e3:,.0f} kW" if rc["active"] else "inactive"
        q_cd = f"Q = {cd['Q']/1e3:,.0f} kW"
        w_t, w_p, w_f = f"{P['W_turb_el']/1e3:,.0f} kW", f"{P['W_pump_el']/1e3:,.1f} kW", f"fans {P['W_fan_el']/1e3:,.0f} kW"
        w_bp = f"brine pump {P['W_geo_pump_el']/1e3:,.1f} kW"
        air_in, air_out = f"{sp.T_air_in_C:.1f} °C · {cd['m_air']:,.0f} kg/s", f"{cd['T_a_out']-273.15:.1f} °C"
        mwf = f"{r['m_wf']:.1f} kg/s {sp.wf}"
    else:
        T = {x: None for x in STATE_LABELS}
        P_hi = P_lo = None
        geo_in = geo_mid = geo_out = q_ev = q_pr = q_rc = q_cd = w_t = w_p = w_f = w_bp = air_in = air_out = mwf = "–"

    def tT(lbl):
        return f"{lbl}  {T[lbl]:.1f} °C" if T[lbl] is not None else f"{lbl}"

    def pbar(p):
        return f"{p:.1f} bar" if p is not None else ""

    box = lambda x, y, w, h: (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                              f'fill="{C_PLANE}" stroke="{C_AXIS}" stroke-width="1.2"/>')
    txt = lambda x, y, s, size=12, anchor="middle", color=C_INK, weight="normal": (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
        f'font-weight="{weight}" font-family="{FONT}">{s}</text>')
    line = lambda pts, color, marker=True: (
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2" '
        f'{"marker-end=\"url(#a_" + color[1:] + ")\"" if marker else ""}/>')
    dot = lambda x, y: f'<circle cx="{x}" cy="{y}" r="4.5" fill="{C_SURF}" stroke="{C_COLD}" stroke-width="2"/>'

    defs = "".join(
        f'<marker id="a_{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in (C_HOT, C_COLD, C_AQUA))

    s = [f'<svg viewBox="0 0 1000 400" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}"><defs>{defs}</defs>']
    # ---- brine (orange): production -> brine pump -> evaporator -> preheater -> injection
    s += [txt(330, 22, "PRODUCTION WELL", 11, color=C_HOT, weight="600"), txt(330, 36, geo_in, 11, color=C_INK2),
          line("330,40 330,68", C_HOT, marker=False),
          f'<circle cx="330" cy="80" r="11" fill="{C_SURF}" stroke="{C_HOT}" stroke-width="2"/>',
          txt(330, 84, "P", 10, color=C_HOT, weight="600"),
          txt(346, 84, w_bp, 9.5, anchor="start", color=C_INK2),
          line("330,91 330,120", C_HOT),
          line("255,120 255,85 125,85 125,120", C_HOT), txt(190, 104, geo_mid, 10, color=C_INK2),
          line("70,120 70,40", C_HOT), txt(70, 22, "INJECTION WELL", 11, color=C_HOT, weight="600"),
          txt(70, 36, geo_out, 11, color=C_INK2)]
    # ---- air (aqua) through the condenser
    s += [line("835,40 835,120", C_AQUA), txt(835, 22, "AMBIENT AIR", 11, color=C_AQUA, weight="600"),
          txt(835, 36, air_in, 11, color=C_INK2),
          line("935,120 935,40", C_AQUA), txt(935, 22, "AIR OUT", 11, color=C_AQUA, weight="600"),
          txt(935, 36, air_out, 11, color=C_INK2)]
    # ---- boxes, top row
    s += [box(30, 120, 140, 80), txt(100, 150, "PREHEATER", 13, weight="600"),
          txt(100, 170, "liquid → bubble point", 9.5, color=C_INK2), txt(100, 188, q_pr, 12)]
    s += [box(245, 120, 140, 80), txt(315, 150, "EVAPORATOR", 13, weight="600"),
          txt(315, 170, "boiling", 9.5, color=C_INK2), txt(315, 188, q_ev, 12)]
    s += [f'<polygon points="440,130 530,115 530,205 440,190" fill="{C_PLANE}" stroke="{C_AXIS}" stroke-width="1.2"/>',
          txt(485, 155, "TURBINE", 13, weight="600"), txt(485, 178, w_t, 12), txt(485, 194, "electric", 10, color=C_INK2),
          f'<line x1="485" y1="115" x2="485" y2="88" stroke="{C_AXIS}" stroke-width="2"/>',
          f'<circle cx="485" cy="72" r="16" fill="{C_SURF}" stroke="{C_AXIS}" stroke-width="1.5"/>',
          txt(485, 77, "G", 13, weight="600")]
    s += [box(590, 120, 140, 80), txt(660, 150, "RECUPERATOR", 13, weight="600"),
          txt(660, 170, "hot side 4 → 4r", 10, color=C_INK2), txt(660, 188, q_rc, 12)]
    s += [box(790, 120, 190, 80), txt(885, 150, "AIR-COOLED CONDENSER", 12, weight="600"),
          txt(885, 170, q_cd, 12), txt(885, 188, w_f, 11, color=C_INK2)]
    # ---- bottom row
    s += [box(590, 280, 140, 70), txt(660, 306, "RECUPERATOR", 13, weight="600"),
          txt(660, 324, "cold side 2 → 2r", 10, color=C_INK2), txt(660, 341, q_rc, 12),
          f'<line x1="660" y1="200" x2="660" y2="280" stroke="{C_AXIS}" stroke-width="1" stroke-dasharray="4 4"/>']
    s += [f'<circle cx="885" cy="315" r="26" fill="{C_PLANE}" stroke="{C_AXIS}" stroke-width="1.2"/>',
          txt(885, 311, "FEED PUMP", 9.5, weight="600"), txt(885, 326, w_p, 10)]
    # ---- working-fluid loop (blue)
    s += [line("170,160 245,160", C_COLD), txt(207, 150, tT("2p"), 10.5), txt(207, 176, "sat. liquid", 9.5, color=C_INK2),
          line("385,160 440,160", C_COLD), txt(412, 150, tT("3"), 10.5), txt(412, 176, pbar(P_hi), 9.5, color=C_INK2),
          line("530,160 590,160", C_COLD), txt(560, 150, tT("4"), 10.5), txt(560, 176, pbar(P_lo), 9.5, color=C_INK2),
          line("730,160 790,160", C_COLD), txt(760, 150, tT("4r"), 10.5),
          line("885,200 885,289", C_COLD), txt(900, 250, tT("1"), 10.5, anchor="start"),
          line("859,315 730,315", C_COLD), txt(795, 305, tT("2"), 10.5), txt(795, 331, pbar(P_hi), 9.5, color=C_INK2),
          line("590,315 100,315 100,200", C_COLD), txt(345, 305, tT("2r"), 10.5),
          txt(345, 331, "liquid → preheater", 9.5, color=C_INK2),
          txt(110, 262, mwf, 10.5, anchor="start", color=C_INK2)]
    for (x, y) in [(207, 160), (412, 160), (560, 160), (760, 160), (885, 245), (795, 315), (345, 315)]:
        s.append(dot(x, y))
    s.append("</svg>")
    return "".join(s)


# =====================================================================
#                          PLOTLY FIGURES
# =====================================================================
def _style(fig, title, xt, yt, height=430, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=C_INK), x=0, xanchor="left"),
        xaxis_title=xt, yaxis_title=yt, height=height,
        margin=dict(l=55, r=20, t=60, b=50),
        font=dict(family=FONT, color=C_INK, size=12),
        plot_bgcolor=C_SURF, paper_bgcolor=C_SURF,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        hovermode="x unified", hoverlabel=dict(bgcolor="white", font=dict(family=FONT, size=12)))
    fig.update_xaxes(gridcolor=C_GRID, zeroline=False, linecolor=C_AXIS, ticks="outside", tickcolor=C_AXIS)
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False, linecolor=C_AXIS, ticks="outside", tickcolor=C_AXIS)
    return fig


def fig_ts(r: dict) -> go.Figure:
    d = ts_diagram_data(r)
    sp = r["spec"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["dome_s_liq"], y=d["dome_T"], mode="lines", name="saturation dome",
                             line=dict(color=C_MUTED, width=1.2), hoverinfo="skip", showlegend=True))
    fig.add_trace(go.Scatter(x=d["dome_s_vap"], y=d["dome_T"], mode="lines", showlegend=False,
                             line=dict(color=C_MUTED, width=1.2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["path_s"], y=d["path_T"], mode="lines", name=f"{sp.wf} cycle",
                             line=dict(color=C_COLD, width=2.2),
                             hovertemplate="s = %{x:.3f} kJ/kgK<br>T = %{y:.1f} °C<extra></extra>"))
    pos = ["bottom right", "bottom left", "top left", "top left", "top right", "bottom right", "bottom right"]
    fig.add_trace(go.Scatter(x=[p["s"] for p in d["points"]], y=[p["T"] for p in d["points"]],
                             mode="markers+text", text=[p["label"] for p in d["points"]],
                             textposition=pos, textfont=dict(size=12, color=C_INK), name="state points",
                             marker=dict(size=9, color=C_SURF, line=dict(color=C_COLD, width=2)),
                             hovertemplate="state %{text}<br>s = %{x:.3f} kJ/kgK<br>T = %{y:.1f} °C<extra></extra>"))
    ss = [p["s"] for p in d["points"]]
    x0, x1 = min(ss) - 0.35, max(ss) + 0.35
    for Tl, c, name in [(d["T_geo_in_C"], C_HOT, "brine in"), (d["T_air_in_C"], C_AQUA, "ambient air")]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=Tl, y1=Tl, line=dict(color=c, width=1.2, dash="dot"))
        fig.add_annotation(x=x1, y=Tl, text=f"{name} {Tl:.0f} °C", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=11, color=c))
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[d["T_air_in_C"] - 10, max(d["T_geo_in_C"], r["dv"].T_evap_C) + 18])
    _style(fig, f"T-s diagram · net {r['power']['W_net']/1e3:,.0f} kW", "specific entropy s [kJ/(kg K)]",
           "temperature [°C]", height=460)
    fig.update_layout(hovermode="closest")
    return fig


def _tq(Q_kW, T_hot, T_cold, hot_name, cold_name, title, min_dT=None, pinch_xy=None, xlabel=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Q_kW, y=T_hot, mode="lines", name=hot_name, line=dict(color=C_HOT, width=2.2),
                             hovertemplate="%{y:.1f} °C"))
    fig.add_trace(go.Scatter(x=Q_kW, y=T_cold, mode="lines", name=cold_name, line=dict(color=C_COLD, width=2.2),
                             hovertemplate="%{y:.1f} °C"))
    if pinch_xy is not None:
        fig.add_trace(go.Scatter(x=[pinch_xy[0]], y=[pinch_xy[1]], mode="markers+text",
                                 text=[f"pinch {min_dT:.1f} K"], textposition="bottom right",
                                 textfont=dict(size=11, color=C_INK2),
                                 marker=dict(size=9, color=C_SURF, line=dict(color=C_INK2, width=2)),
                                 name="pinch", hoverinfo="skip", showlegend=False))
    _style(fig, title, xlabel or "heat transferred Q [kW]  (from cold end)", "temperature [°C]", height=400)
    fig.update_xaxes(hoverformat=",.0f")
    return fig


def fig_evaporator(r):
    ev = r["evaporator"]; sp = r["spec"]
    Q = ev["tq_Q"] / 1e3
    i = int(np.argmin(ev["tq_T_h"] - ev["tq_T_c"]))
    return _tq(Q, ev["tq_T_h"] - 273.15, ev["tq_T_c"] - 273.15, f"brine ({sp.geo_fluid})", f"{sp.wf} (2p → 3)",
               f"Evaporator T-Q · {ev['Q']/1e3:,.0f} kW · pinch {ev['min_dT']:.1f} K",
               ev["min_dT"], (Q[i], ev["tq_T_c"][i] - 273.15), "heat transferred Q [kW]  (from brine outlet end)")


def fig_preheater(r):
    pr = r["preheater"]; sp = r["spec"]
    Q = pr["tq_Q"] / 1e3
    i = int(np.argmin(pr["tq_T_h"] - pr["tq_T_c"]))
    return _tq(Q, pr["tq_T_h"] - 273.15, pr["tq_T_c"] - 273.15, f"brine ({sp.geo_fluid})", f"{sp.wf} (2r → 2p)",
               f"Preheater T-Q · {pr['Q']/1e3:,.0f} kW · pinch {pr['min_dT']:.1f} K",
               pr["min_dT"], (Q[i], pr["tq_T_c"][i] - 273.15), "heat transferred Q [kW]  (from injection end)")


def fig_train(r):
    t = heat_train_data(r); sp = r["spec"]; pr, ev = r["preheater"], r["evaporator"]
    fig = _tq(t["Q"], t["T_hot"], t["T_cold"], f"brine ({sp.geo_fluid})", sp.wf,
              f"Evaporator + preheater, brine side · preheater {pr['Q']/1e3:,.0f} kW + evaporator {ev['Q']/1e3:,.0f} kW",
              xlabel="cumulative heat from the brine Q [kW]  (from injection end)")
    y_top = max(t["T_hot"]) + 2
    fig.add_shape(type="line", x0=t["Q_split_kW"], x1=t["Q_split_kW"], y0=min(t["T_cold"]) - 2, y1=y_top,
                  line=dict(color=C_MUTED, width=1, dash="dash"))
    fig.add_annotation(x=t["Q_split_kW"] / 2, y=y_top, text="preheater", showarrow=False, font=dict(size=11, color=C_INK2))
    fig.add_annotation(x=(t["Q_split_kW"] + t["Q"][-1]) / 2, y=y_top, text="evaporator", showarrow=False,
                       font=dict(size=11, color=C_INK2))
    return fig


def fig_recuperator(r):
    rc = r["recuperator"]
    if not rc["active"]:
        return None
    Q = rc["tq_Q"] * r["m_wf"] / 1e3
    i = int(np.argmin(rc["tq_T_h"] - rc["tq_T_c"]))
    return _tq(Q, rc["tq_T_h"] - 273.15, rc["tq_T_c"] - 273.15, "turbine exhaust 4 → 4r", "pump discharge 2 → 2r",
               f"Recuperator T-Q · {rc['Q']/1e3:,.0f} kW · effectiveness {rc['effectiveness']:.2f} · pinch {rc['min_dT']:.1f} K",
               rc["min_dT"], (Q[i], rc["tq_T_c"][i] - 273.15))


def fig_condenser(r):
    c = condenser_tq_data(r); cd = r["condenser"]; sp = r["spec"]
    return _tq(c["Q"] / 1e3, c["T_hot"], c["T_cold"], f"{sp.wf} 4r → 1", f"air · {cd['m_air']:,.0f} kg/s",
               f"Air-cooled condenser T-Q · {cd['Q']/1e3:,.0f} kW · pinch {cd['min_dT']:.1f} K · fans {cd['W_fan']/1e3:,.0f} kW",
               cd["min_dT"], (c["Q"][2] / 1e3, c["T_cold"][2]), "heat transferred Q [kW]  (from condensate / air-inlet end)")


def fig_map(opt: dict, r: dict | None) -> go.Figure | None:
    W = opt["W_map"]
    if np.all(np.isnan(W)):
        return None
    floor = -100.0
    Wp = np.where(np.isnan(W), np.nan, np.clip(W, floor, None))
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=opt["T_evap_grid"], y=opt["T_cond_grid"], z=Wp,
        colorscale=[(i / (len(BLUES) - 1), c) for i, c in enumerate(BLUES)],
        ncontours=22, contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
        line=dict(width=0.5, color="rgba(255,255,255,0.6)"),
        colorbar=dict(title=dict(text="net power [kW]", side="right"), thickness=14, len=0.9),
        hovertemplate="T_evap %{x:.0f} °C · T_cond %{y:.0f} °C<br>net %{z:,.0f} kW<extra></extra>",
        connectgaps=False, name="net power"))
    fig.add_trace(go.Contour(
        x=opt["T_evap_grid"], y=opt["T_cond_grid"], z=W, showscale=False, hoverinfo="skip",
        contours=dict(start=0, end=0, size=1, coloring="none"),
        line=dict(color=C_INK2, width=1.5, dash="dash"), name="net = 0", showlegend=True))
    if r is not None and opt.get("feasible"):
        x = opt["x_opt"]
        fig.add_trace(go.Scatter(x=[x[0]], y=[x[2]], mode="markers", name=f"optimum {r['power']['W_net']/1e3:,.0f} kW",
                                 marker=dict(symbol="star", size=18, color=C_HOT, line=dict(color="white", width=1.2)),
                                 hovertemplate=f"optimum<br>T_evap {x[0]:.1f} °C · T_cond {x[2]:.1f} °C<extra></extra>"))
    held = opt.get("superheat_held")
    sh_txt = "superheat optimised after the grid" if held is None else f"superheat {held:.1f} K"
    _style(fig, f"Net power map ({sh_txt}) · blank = infeasible, clipped below {floor:.0f} kW",
           "evaporation temperature T_evap [°C]", "condensing temperature T_cond [°C]", height=520)
    fig.update_layout(hovermode="closest")
    return fig


# =====================================================================
#                          RESULT RENDERING
# =====================================================================
def banners(r: dict, design: dict):
    sp = r["spec"]; ck = r["checks"]; rc = r["recuperator"]; br = r["brine"]
    rj = ck["reinjection"]
    b1, b2 = br["states"][1], br["states"][2]
    path = (f"brine {sp.T_geo_in_C:.1f} °C → evaporator → {b1['T_C']:.1f} °C ({b1['P_bar']:.1f} bar) → "
            f"preheater → **{b2['T_C']:.1f} °C at {b2['P_bar']:.1f} bar** to injection")
    if rj["mode"] == "free":
        st.info(f"**Reinjection** {path} — pinch-limited (no constraint applied). "
                f"Working-fluid flow set by the {br['limiting']}.", icon="💧")
    else:
        st.success(f"**Reinjection** {path} — inside the allowed {rj['floor_C']:.1f} … {rj['ceiling_C']:.1f} °C "
                   f"(target {rj['target_C']:.0f} °C, deviation {rj['deviation_K']:+.1f} K). "
                   f"Working-fluid flow set by the {br['limiting']}.", icon="💧")
    lq = ck["liquid_at_preheater"]
    st.success(f"**Liquid before the preheater** — state 2r is subcooled by {lq['subcool_K']:.1f} K "
               f"(minimum {lq['min_K']:.1f} K); the preheater delivers saturated liquid at "
               f"{r['dv'].T_evap_C:.1f} °C to the evaporator"
               + (" · recuperator duty capped to keep 2r liquid" if lq["capped"] else "") + ".", icon="✅")
    pe, ppn, pr_, pc = ck["pinch"]["evaporator"], ck["pinch"]["preheater"], ck["pinch"]["recuperator"], ck["pinch"]["condenser"]
    st.info(f"**Minimum approach** — evaporator {pe[0]:.1f} K (target {pe[1]:.0f} K, at the "
            f"{r['evaporator']['pinch_location']}) · preheater {ppn[0]:.1f} K (target {ppn[1]:.0f} K, at the "
            f"{r['preheater']['pinch_location']}) · recuperator "
            f"{(f'{pr_[0]:.1f} K' if pr_[0] is not None else '–')} (target {pr_[1]:.0f} K) · condenser "
            f"{pc[0]:.1f} K (target {pc[1]:.0f} K).", icon="🌡️")
    if sp.recuperator and not rc["active"]:
        S = r["states"]; k = list(S)
        why = (f"capped to zero by the {rc['cap_reason']}" if rc["capped"] else
               f"the turbine exhaust ({S[k[5]]['T_C']:.1f} °C) is not {sp.pinch_rec_K:.0f} K above the pump "
               f"discharge ({S[k[1]]['T_C']:.1f} °C), so a {sp.pinch_rec_K:.0f} K pinch leaves nothing to recover")
        st.warning(f"**Recuperator inactive** — {why}.", icon="⚠️")
    elif rc["active"] and rc["capped"]:
        st.warning(f"**Recuperator duty capped** by the {rc['cap_reason']} (pinch {rc['min_dT']:.1f} K "
                   f"> target {sp.pinch_rec_K:.0f} K).", icon="⚠️")
    if not ck["dry_expansion"]:
        st.warning(f"**Wet expansion** — turbine exhaust quality x = {ck['turbine_outlet_quality']:.3f}. "
                   "Consider a drier fluid or superheat.", icon="⚠️")
    if design.get("opt") is not None:
        o = design["opt"]
        st.caption(f"Optimised in {o['n_eval']} cycle evaluations ({o['seconds']:.0f} s) · run {design['when']}")
    else:
        st.caption(f"Manual design point · run {design['when']}")


def kpis(r: dict):
    P, pf, dv, cd = r["power"], r["perf"], r["dv"], r["condenser"]
    c = st.columns([2, 1, 1, 1, 1])
    with c[0]:
        st.markdown(f'<div class="hero">{P["W_net"]/1e3:,.0f} kW</div>'
                    f'<div class="hero-sub">NET ELECTRIC POWER</div>', unsafe_allow_html=True)
    c[1].metric("Turbine", f"{P['W_turb_el']/1e3:,.0f} kW", help="electric, after mechanical and generator losses")
    c[2].metric("Fans", f"−{P['W_fan_el']/1e3:,.0f} kW", help="air-cooled condenser fans (electric)")
    c[3].metric("Feed pump", f"−{P['W_pump_el']/1e3:,.1f} kW", help="working-fluid pump, electric")
    c[4].metric("Brine pump", f"−{P['W_geo_pump_el']/1e3:,.1f} kW",
                help=f"restores {r['brine']['dP_total_bar']:.2f} bar lost by the brine")
    d = st.columns(6)
    d[0].metric("WF flow", f"{r['m_wf']:.1f} kg/s", help="working-fluid mass flow")
    d[1].metric("Reinjection", f"{pf['T_geo_out_C']:.1f} °C", f"{pf['P_geo_out_bar']:.2f} bar", delta_color="off",
                help="brine temperature and pressure to the injection well")
    d[2].metric("Evaporation", f"{dv.T_evap_C:.1f} °C", f"{r['P_evap_bar']:.2f} bar", delta_color="off")
    d[3].metric("Superheat", f"{dv.dT_sh_K:.1f} K", f"inlet {dv.T_evap_C + dv.dT_sh_K:.1f} °C", delta_color="off")
    d[4].metric("Condensing", f"{dv.T_cond_C:.1f} °C", f"{r['P_cond_bar']:.2f} bar", delta_color="off")
    d[5].metric("Pressure ratio", f"{r['turbine']['PR']:.2f}")
    e = st.columns(6)
    e[0].metric("Heat input", f"{pf['Q_in']/1e3:,.0f} kW",
                f"evap {pf['Q_evap']/1e3:,.0f} + pre {pf['Q_pre']/1e3:,.0f}", delta_color="off",
                help=f"heat taken from the brine; {100*pf['utilisation']:.0f} % of the heat above ambient")
    e[1].metric("Recuperated", f"{pf['Q_rec']/1e3:,.0f} kW", help="heat recovered from the turbine exhaust")
    e[2].metric("Net efficiency", f"{100*pf['eta_net']:.2f} %", help="W_net / heat input")
    e[3].metric("Thermal eff.", f"{100*pf['eta_th']:.2f} %", help="(turbine − feed pump shaft work) / heat input")
    e[4].metric("Exergy eff.", f"{100*pf['eta_II']:.1f} %", help="W_net / brine exergy at ambient dead state")
    e[5].metric("Specific power", f"{pf['W_net_per_kg_geo']/1e3:.2f} kJ/kg",
                help="net power per kg/s of brine (kW per kg/s = kJ/kg)")


def tables(r: dict):
    sp = r["spec"]
    rows = []
    for name, s in r["states"].items():
        rows.append(dict(point=name, T_C=s["T_C"], P_bar=s["P_bar"], h_kJkg=s["h_kJkg"], s_kJkgK=s["s_kJkgK"],
                         quality=(f"{s['x']:.3f}" if (s["x"] is not None and 0 <= s["x"] <= 1) else "–")))
    states = pd.DataFrame(rows).rename(columns={"T_C": "T [°C]", "P_bar": "P [bar]", "h_kJkg": "h [kJ/kg]",
                                                "s_kJkgK": "s [kJ/kgK]", "quality": "x [-]"})
    brine = pd.DataFrame(r["brine"]["states"]).rename(columns={"T_C": "T [°C]", "P_bar": "P [bar]",
                                                               "h_kJkg": "h [kJ/kg]"})
    ev, pr, rc, cd = r["evaporator"], r["preheater"], r["recuperator"], r["condenser"]
    hx = pd.DataFrame([
        dict(exchanger="Evaporator", Q_kW=ev["Q"] / 1e3, min_dT_K=f"{ev['min_dT']:.2f}", pinch_target_K=sp.pinch_evap_K,
             A_m2=ev["A"], UA_kW_K=sp.U_evap * ev["A"] / 1e3,
             notes=f"brine {sp.T_geo_in_C:.1f} → {ev['T_geo_out']-273.15:.1f} °C ({ev['P_geo_in_bar']:.1f} → "
                   f"{ev['P_geo_out_bar']:.1f} bar) · boil {ev['Q_boil']/1e3:.0f} / superheat {ev['Q_sh']/1e3:.0f} kW · "
                   f"pinch at {ev['pinch_location']}"),
        dict(exchanger="Preheater", Q_kW=pr["Q"] / 1e3, min_dT_K=f"{pr['min_dT']:.2f}", pinch_target_K=sp.pinch_pre_K,
             A_m2=pr["A"], UA_kW_K=sp.U_pre * pr["A"] / 1e3,
             notes=f"brine {pr['T_geo_in']-273.15:.1f} → {pr['T_geo_out']-273.15:.1f} °C ({pr['P_geo_in_bar']:.1f} → "
                   f"{pr['P_geo_out_bar']:.1f} bar) · wf to bubble point {r['dv'].T_evap_C:.1f} °C · "
                   f"pinch at {pr['pinch_location']}"),
        dict(exchanger="Recuperator", Q_kW=rc["Q"] / 1e3, min_dT_K=(f"{rc['min_dT']:.2f}" if rc["active"] else "–"),
             pinch_target_K=sp.pinch_rec_K, A_m2=rc["A"], UA_kW_K=sp.U_rec * rc["A"] / 1e3,
             notes=(f"effectiveness {rc['effectiveness']:.3f}" + (f" · capped by {rc['cap_reason']}" if rc["capped"] else "")
                    if rc["active"] else ("not installed" if not sp.recuperator else "inactive"))),
        dict(exchanger="Air-cooled condenser", Q_kW=cd["Q"] / 1e3, min_dT_K=f"{cd['min_dT']:.2f}", pinch_target_K=sp.pinch_cond_K,
             A_m2=cd["A"], UA_kW_K=cd["U_eff"] * cd["A"] / 1e3,
             notes=f"desuperheat {cd['Q_dsh']/1e3:.0f} / condense {cd['Q_cnd']/1e3:.0f} / subcool {cd['Q_sub']/1e3:.0f} kW · "
                   f"air {cd['m_air']:,.0f} kg/s ({cd['V_air']:,.0f} m³/s), {sp.T_air_in_C:.1f} → {cd['T_a_out']-273.15:.1f} °C · ITD {cd['ITD']:.1f} K"),
    ]).rename(columns={"Q_kW": "Q [kW]", "min_dT_K": "min ΔT [K]", "pinch_target_K": "pinch target [K]",
                       "A_m2": "area [m²]", "UA_kW_K": "UA [kW/K]"})
    P, pf, bp = r["power"], r["perf"], r["brine"]["pump"]
    power = pd.DataFrame([
        ("Turbine shaft power", P["W_turb_shaft"] / 1e3), ("Turbine electric power", P["W_turb_el"] / 1e3),
        ("Feed pump shaft power", P["W_pump_shaft"] / 1e3), ("Feed pump electric power", P["W_pump_el"] / 1e3),
        ("Condenser fan power (electric)", P["W_fan_el"] / 1e3),
        (f"Brine pump electric power ({bp['dP_bar']:.2f} bar)", P["W_geo_pump_el"] / 1e3),
        ("NET ELECTRIC POWER", P["W_net"] / 1e3),
        ("Heat from brine, evaporator", pf["Q_evap"] / 1e3), ("Heat from brine, preheater", pf["Q_pre"] / 1e3),
        ("Recuperated heat", pf["Q_rec"] / 1e3), ("Heat rejected (condenser)", pf["Q_cond"] / 1e3)],
        columns=["quantity", "kW"])
    perf = pd.DataFrame([
        ("Cycle thermal efficiency (shaft)", f"{100*pf['eta_th']:.2f} %"),
        ("Gross electric efficiency", f"{100*pf['eta_gross_el']:.2f} %"),
        ("Net electric efficiency", f"{100*pf['eta_net']:.2f} %"),
        ("Exergy (2nd-law) efficiency", f"{100*pf['eta_II']:.2f} %  (brine exergy {pf['Ex_geo_in']/1e3:,.0f} kW at {sp.T_air_in_C:.0f} °C)"),
        ("Heat-source utilisation", f"{100*pf['utilisation']:.1f} %  (of the heat above ambient)"),
        ("Back-work ratio", f"{100*pf['back_work_ratio']:.1f} %  ((pumps + fans) / turbine)"),
        ("Specific net power", f"{pf['W_net_per_kg_geo']/1e3:.2f} kW per kg/s brine"),
        ("Brine after evaporator", f"{pf['T_geo_mid_C']:.2f} °C"),
        ("Brine to injection", f"{pf['T_geo_out_C']:.2f} °C at {pf['P_geo_out_bar']:.2f} bar"),
        ("Feed-pump temperature rise", f"{r['pump']['dT']:.2f} K"),
        ("Turbine isentropic enthalpy drop", f"{r['turbine']['dh_isen']/1e3:.2f} kJ/kg"),
        ("Turbine outlet", f"{r['turbine']['phase_out']}" + (f", x = {r['turbine']['x_out']:.3f}" if r["checks"]["turbine_outlet_quality"] is not None and 0 <= r["checks"]["turbine_outlet_quality"] <= 1 else ""))],
        columns=["metric", "value"])
    return states, brine, hx, power, perf


def profiles_csv(r: dict) -> str:
    ev, pr, rc = r["evaporator"], r["preheater"], r["recuperator"]
    c = condenser_tq_data(r)
    parts = [pd.DataFrame(dict(exchanger="preheater", Q_kW=pr["tq_Q"] / 1e3, T_hot_C=pr["tq_T_h"] - 273.15,
                               T_cold_C=pr["tq_T_c"] - 273.15)),
             pd.DataFrame(dict(exchanger="evaporator", Q_kW=ev["tq_Q"] / 1e3, T_hot_C=ev["tq_T_h"] - 273.15,
                               T_cold_C=ev["tq_T_c"] - 273.15))]
    if rc["active"]:
        parts.append(pd.DataFrame(dict(exchanger="recuperator", Q_kW=rc["tq_Q"] * r["m_wf"] / 1e3,
                                       T_hot_C=rc["tq_T_h"] - 273.15, T_cold_C=rc["tq_T_c"] - 273.15)))
    parts.append(pd.DataFrame(dict(exchanger="condenser", Q_kW=c["Q"] / 1e3, T_hot_C=c["T_hot"], T_cold_C=c["T_cold"])))
    return pd.concat(parts).to_csv(index=False)


def render(design: dict):
    r = design["result"]
    if r is None:
        st.error(design["error"], icon="🚫")
        info = design.get("info") or {}
        tips = []
        if "T_geo_out_C" in info:
            tips.append(f"The pinch only allows cooling the brine to about **{info['T_geo_out_C']:.1f} °C** there; "
                        "widen the reinjection tolerance, raise the target, or lower the evaporator/preheater pinch.")
        if "flash" not in design["error"]:
            tips += ["Check that the condensing temperature can sit above ambient + condenser pinch.",
                     "A lower evaporation temperature extracts more heat (cools the brine further) at lower efficiency."]
        if tips:
            st.markdown("**Suggestions**\n" + "\n".join(f"- {t}" for t in tips))
        st.markdown(plant_svg(None), unsafe_allow_html=True)
        return

    kpis(r)
    st.markdown("")
    banners(r, design)
    st.markdown(plant_svg(r), unsafe_allow_html=True)

    tabs = st.tabs(["Diagrams", "State points", "Heat exchangers", "Power & performance",
                    "Optimisation map", "Verification", "Export"])
    states, brine, hx, power, perf = tables(r)

    with tabs[0]:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.plotly_chart(fig_ts(r), **WIDE)
        with c2:
            st.plotly_chart(fig_train(r), **WIDE)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(fig_evaporator(r), **WIDE)
        with c4:
            st.plotly_chart(fig_preheater(r), **WIDE)
        c5, c6 = st.columns(2)
        with c5:
            f = fig_recuperator(r)
            if f is None:
                st.info("Recuperator inactive at this design point — no T-Q profile.", icon="ℹ️")
            else:
                st.plotly_chart(f, **WIDE)
        with c6:
            st.plotly_chart(fig_condenser(r), **WIDE)

    with tabs[1]:
        st.markdown("**Working fluid**")
        st.dataframe(states.style.format({"T [°C]": "{:.2f}", "P [bar]": "{:.2f}", "h [kJ/kg]": "{:.2f}",
                                          "s [kJ/kgK]": "{:.4f}"}),
                     **WIDE, hide_index=True)
        lq = r["checks"]["liquid_at_preheater"]
        st.caption(f"State 2r (preheater inlet) is subcooled liquid by {lq['subcool_K']:.2f} K "
                   f"(minimum {lq['min_K']:.2f} K); 2p is saturated liquid (x = 0). "
                   "Working-fluid pressures are constant on each side (no pressure drops).")
        st.markdown("**Brine**")
        st.dataframe(brine.style.format({"T [°C]": "{:.2f}", "P [bar]": "{:.2f}", "h [kJ/kg]": "{:.2f}"}),
                     **WIDE, hide_index=True)

    with tabs[2]:
        st.dataframe(hx.style.format({"Q [kW]": "{:,.1f}", "pinch target [K]": "{:.1f}",
                                      "area [m²]": "{:,.1f}", "UA [kW/K]": "{:,.1f}"}),
                     **WIDE, hide_index=True)
        cd = r["condenser"]
        st.markdown(f"**Condenser air side** — {cd['m_air']:,.0f} kg/s ({cd['V_air']:,.0f} m³/s) at "
                    f"{cd['rho_air']:.3f} kg/m³; air {cd['T_a_in']-273.15:.1f} → after subcool {cd['T_a_b']-273.15:.1f} → "
                    f"after condensing {cd['T_a_c']-273.15:.1f} → out {cd['T_a_out']-273.15:.1f} °C; "
                    f"zone areas desuperheat {cd['A_dsh']:.0f} / condense {cd['A_cnd']:.0f} / subcool {cd['A_sub']:.0f} m²; "
                    f"fan power {cd['W_fan']/1e3:.1f} kW.")
        st.caption("Areas use the U values in the sidebar and only report sizing; they do not affect the power.")

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Power balance**")
            st.dataframe(power.style.format({"kW": "{:,.1f}"}), **WIDE, hide_index=True)
        with c2:
            st.markdown("**Performance**")
            st.dataframe(perf, **WIDE, hide_index=True)

    with tabs[4]:
        opt = design.get("opt")
        if opt is None:
            st.info("Run in *Optimise for maximum net power* mode to see the map.", icon="ℹ️")
        else:
            f = fig_map(opt, r)
            if f is None:
                st.warning("No feasible point on the grid.")
            else:
                st.plotly_chart(f, **WIDE)
                (Te, sh, Tc) = design_bounds(r["spec"])
                st.caption(f"Search space: T_evap {Te[0]:.1f}–{Te[1]:.1f} °C, T_cond {Tc[0]:.1f}–{Tc[1]:.1f} °C"
                           + (f", superheat {sh[0]:.1f}–{sh[1]:.0f} K" if opt.get("superheat_held") is None else "")
                           + f" in {GRID_STEP_K:.0f} K steps. The star is the refined optimum.")

    with tabs[5]:
        st.markdown("Each exchanger is re-rated at its design area with the **rating** models of the component "
                    "library (`HEX.hex_rate`, `AirCooledCondenser.air_cooled_condenser`); the duty should be reproduced.")
        rows = []
        for v in verify_with_rating_models(r):
            dev = (np.nan if np.isnan(v["Q_rating"]) else 100 * (v["Q_rating"] - v["Q_design"]) / v["Q_design"])
            rows.append(dict(model=v["name"], Q_design_kW=v["Q_design"] / 1e3, Q_rating_kW=v["Q_rating"] / 1e3,
                             deviation_pct=dev, notes=v["extra"]))
        st.dataframe(pd.DataFrame(rows).style.format({"Q_design_kW": "{:,.1f}", "Q_rating_kW": "{:,.1f}",
                                                       "deviation_pct": "{:+.3f}"}, na_rep="–"),
                     **WIDE, hide_index=True)

    with tabs[6]:
        sp = r["spec"]
        tag = f"{sp.wf}_{sp.T_geo_in_C:.0f}C_{sp.geo_outlet}"
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("Design summary (JSON)", json.dumps(result_to_dict(r), indent=2, default=_json_default),
                           f"SimRORC_design_{tag}.json", "application/json", **WIDE)
        c2.download_button("State points, WF + brine (CSV)",
                           states.to_csv(index=False) + "\n" + brine.to_csv(index=False),
                           f"SimRORC_states_{tag}.csv", "text/csv", **WIDE)
        c3.download_button("Heat exchangers (CSV)", hx.to_csv(index=False), f"SimRORC_exchangers_{tag}.csv", "text/csv",
                           **WIDE)
        c4.download_button("T-Q profiles (CSV)", profiles_csv(r), f"SimRORC_TQ_profiles_{tag}.csv", "text/csv",
                           **WIDE)
        st.download_button("Inputs (JSON) — reproduce this run from the command line", json.dumps(asdict(sp), indent=2),
                           f"SimRORC_inputs_{tag}.json", "application/json")
        st.caption("The same model runs headless: `python ORC_recuperated.py --help`. Use `PlantSpec(**inputs)` "
                   "with `optimise_plant` / `solve_cycle` to embed it in a bigger model.")


# =====================================================================
#                                MAIN
# =====================================================================
def main():
    spec, ctl = sidebar()
    st.markdown(f"# {APP_NAME}")
    st.markdown(f'<div class="byline">by <b>{AUTHOR}</b> · <a href="{LINKEDIN}" target="_blank">'
                f'www.linkedin.com/in/md-faisal-karim</a></div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Design mode · recuperated organic Rankine cycle on a liquid geothermal or '
                'waste-heat source · brine through evaporator then preheater (pressure drop restored by a brine pump) · '
                'preheater to the bubble point · liquid guaranteed before the preheater · pinch-limited exchangers · '
                'air-cooled condenser · reinjection target with tolerance</div>', unsafe_allow_html=True)
    st.markdown("")

    if ctl["run"]:
        st.session_state["design"] = compute(spec, ctl)

    design = st.session_state.get("design")
    if design is None:
        st.info("Set the inputs in the sidebar and press **Run design**. The optimiser searches the evaporation and "
                "condensing temperatures (and superheat, if enabled) for the highest net electric power that respects "
                "every pinch, keeps the working fluid liquid before the preheater and returns the brine inside the "
                "reinjection band.", icon="👈")
        st.markdown(plant_svg(None), unsafe_allow_html=True)
    else:
        render(design)

    with st.expander("Method & assumptions"):
        st.markdown("""
* **Brine train** — the brine enters the **evaporator** first (boiling the working fluid from its bubble point) and then
  the **preheater** (heating the liquid from the recuperator outlet exactly to the bubble point); it loses the set
  pressure drop in each exchanger and a brine pump restores the loop pressure (auxiliary load in the net power).
* **Components** — turbine (isentropic + mechanical efficiency), feed pump (isentropic efficiency, liquid inlet),
  counter-flow exchangers (T-Q profile with phase-boundary refinement, pinch and area), air-cooled condenser
  (3-zone desuperheat / condense / subcool with LMTD, fan power = volumetric flow × Δp / η). Fluid properties: CoolProp.
* **Working-fluid flow** — the largest flow that respects the evaporator and preheater pinches on the full profiles
  without cooling the brine below the reinjection floor; a design point that cannot cool the brine to the
  reinjection ceiling is rejected.
* **Recuperator** — maximum recovery at its pinch, capped so that (i) state 2r stays liquid with the requested
  subcooling and (ii) the preheater cold end keeps its pinch when the reinjection ceiling binds.
* **Condenser** — the smallest air flow that respects the condenser pinch (the approach between the condensing
  temperature and the air leaving the condensing zone); the condensing temperature is optimised against fan power.
* **Optimiser** — grid over (T_evap, T_cond) in 2 K steps, then Nelder-Mead refinement (superheat included only in
  *Optimise* mode). Net power = turbine electric − feed pump − fans − brine pump.
* **Not modelled** — working-fluid pressure drops, heat losses, part load, generator cooling, other plant auxiliaries.
""")
    st.markdown(f'<div class="footer">{APP_NAME} · by {AUTHOR} · '
                f'<a href="{LINKEDIN}" target="_blank">www.linkedin.com/in/md-faisal-karim</a></div>',
                unsafe_allow_html=True)


main()
