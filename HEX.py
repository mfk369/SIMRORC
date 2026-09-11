"""
counter-flow heat exchanger - rating mode.

Given hot/cold fluid identities, inlet states, mass flows, area, U, and a
minimum-pinch target, return the operating point: heat duty, outlet
states, achieved pinch, required area, and the full T-Q profile.

Auothor: fkarim
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _lmtd(dT1: float, dT2: float) -> float:
    dT1 = max(dT1, 1e-4)
    dT2 = max(dT2, 1e-4)
    if abs(dT1 - dT2) < 1e-6:
        return 0.5 * (dT1 + dT2)
    return (dT2 - dT1) / np.log(dT2 / dT1)


def _phase_boundary_enthalpies(P, fluid):
    """Return (h_bub, h_dew) at pressure P if both branches exist, else None."""
    try:
        P_crit = PropsSI("Pcrit", fluid)
        if P >= P_crit:
            return None
        h_b = PropsSI("H", "P", P, "Q", 0, fluid)
        h_d = PropsSI("H", "P", P, "Q", 1, fluid)
        return h_b, h_d
    except Exception:
        return None


def _tq_profile(h_co, *, h_ci, h_hi, P_h, P_c, m_h, m_c,
                hot_fluid, cold_fluid, U, n_seg):
    """Build a TQ profile assuming a given cold-side outlet enthalpy.

    Counter-flow indexing:
        index 0   = cold-inlet  / hot-outlet end
        index n   = cold-outlet / hot-inlet  end

    The grid is enriched with extra points at any saturation boundary the
    cold OR hot stream crosses, so cp discontinuities at h_bub / h_dew are
    resolved exactly (no smearing into superheat / 2-phase neighbours).
    """
    Q_total = m_c * (h_co - h_ci)

    # Uniform cold-side grid as the base
    h_c = list(np.linspace(h_ci, h_co, n_seg + 1))

    # Insert extra h_c points wherever the COLD fluid crosses h_bub or h_dew
    cold_sat = _phase_boundary_enthalpies(P_c, cold_fluid)
    if cold_sat is not None:
        for h_sat in cold_sat:
            lo, hi = min(h_ci, h_co), max(h_ci, h_co)
            if lo < h_sat < hi:
                # add 4 points tightly bracketing the discontinuity
                for eps in (-50.0, -1.0, +1.0, +50.0):
                    h_c.append(h_sat + eps)

    # Insert extra h_c points wherever the HOT fluid crosses its h_bub/h_dew.
    # Each cold-side h_c maps to a hot-side h via energy balance, so we map
    # the hot saturation enthalpies back to a cold-side h.
    hot_sat = _phase_boundary_enthalpies(P_h, hot_fluid)
    if hot_sat is not None:
        # h_h = h_hi - (Q_total - m_c*(h_c - h_ci))/m_h
        # solve for h_c that gives h_h = h_sat_hot:
        #   h_hi - h_sat_hot = (Q_total - m_c*(h_c - h_ci))/m_h
        #   m_h*(h_hi - h_sat_hot) = Q_total - m_c*(h_c - h_ci)
        #   h_c = h_ci + (Q_total - m_h*(h_hi - h_sat_hot))/m_c
        for h_sat_h in hot_sat:
            h_c_match = h_ci + (Q_total - m_h * (h_hi - h_sat_h)) / m_c
            lo, hi = min(h_ci, h_co), max(h_ci, h_co)
            if lo < h_c_match < hi:
                for eps in (-50.0, -1.0, +1.0, +50.0):
                    h_c.append(h_c_match + eps)

    h_c = np.array(sorted(set(round(v, 3) for v in h_c)))
    Q_arr = m_c * (h_c - h_ci)
    h_h = h_hi - (Q_total - Q_arr) / m_h

    # Vectorised CoolProp call: one C-level call per stream instead of a
    # Python loop over segments. Bit-identical to the per-segment loop
    # but ~10x faster because we eliminate the Python<->C marshalling per
    # element. CoolProp's PropsSI accepts numpy arrays and broadcasts P.
    T_h = np.asarray(PropsSI("T", "H", np.ascontiguousarray(h_h),
                               "P", P_h, hot_fluid))
    T_c = np.asarray(PropsSI("T", "H", np.ascontiguousarray(h_c),
                               "P", P_c, cold_fluid))

    dT_all = T_h - T_c
    min_dT = float(dT_all.min())

    n = len(h_c) - 1
    dQ = np.diff(Q_arr)
    A_seg = np.empty(n)
    for i in range(n):
        lm = _lmtd(dT_all[i], dT_all[i + 1])
        A_seg[i] = dQ[i] / (U * lm) if lm > 0 else np.inf
    A_req = float(A_seg.sum())

    return Q_total, T_h, T_c, Q_arr, A_req, min_dT


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def hex_rate(*, hot_fluid, P_h_bar, m_h, h_h_in_Jkg=None, T_h_in_C=None,
             cold_fluid, P_c_bar, m_c, h_c_in_Jkg=None, T_c_in_C=None,
             A_m2, U_W_m2K, pinch_K=1.0, n_seg=80,
             T_h_out_min_C=None,
             name="HEX"):
    """Rate a counter-flow HEX.

    Inlet states may be specified either by enthalpy (preferred — avoids
    saturation-line ambiguities) or by temperature.

    Operating point = whichever of (area binds, pinch binds) is reached
    first as the cold-side outlet enthalpy is raised from h_ci upward.
    """
    P_h = P_h_bar * 1e5
    P_c = P_c_bar * 1e5

    if h_h_in_Jkg is not None:
        h_hi = h_h_in_Jkg
        T_hi = PropsSI("T", "H", h_hi, "P", P_h, hot_fluid)
    else:
        T_hi = T_h_in_C + 273.15
        h_hi = PropsSI("H", "T", T_hi, "P", P_h, hot_fluid)

    if h_c_in_Jkg is not None:
        h_ci = h_c_in_Jkg
        T_ci = PropsSI("T", "H", h_ci, "P", P_c, cold_fluid)
    else:
        T_ci = T_c_in_C + 273.15
        h_ci = PropsSI("H", "T", T_ci, "P", P_c, cold_fluid)

    T_h_in_C = T_hi - 273.15
    T_c_in_C = T_ci - 273.15

    if T_hi <= T_ci + pinch_K:
        # No driving force - HEX does nothing
        return dict(
            name=name, Q=0.0, A_required=0.0, A_installed=A_m2,
            T_h_in_C=T_h_in_C, T_h_out_C=T_h_in_C,
            T_c_in_C=T_c_in_C, T_c_out_C=T_c_in_C,
            P_h_bar=P_h_bar, P_c_bar=P_c_bar,
            m_h=m_h, m_c=m_c,
            h_h_in=h_hi, h_h_out=h_hi, h_c_in=h_ci, h_c_out=h_ci,
            x_c_out=None, x_h_out=None,
            pinch_K_achieved=T_hi - T_ci, pinch_K_target=pinch_K,
            limiting="no-driving-force",
            U_W_m2K=U_W_m2K,
            tq_T_h=None, tq_T_c=None, tq_Q=None,
        )

    # Cold-side cap: T_c_out cannot exceed T_h_in - pinch_K
    T_co_max = T_hi - pinch_K
    h_co_max_cold = PropsSI("H", "T", T_co_max, "P", P_c, cold_fluid)

    # Hot-side cap: T_h_out cannot fall below T_c_in + pinch_K.
    # This also keeps hot enthalpy inside CoolProp's valid range when
    # m_c >> m_h or the cold side demands more heat than the hot stream
    # can physically deliver before freezing.
    T_ho_min = T_ci + pinch_K
    h_ho_min = PropsSI("H", "T", T_ho_min, "P", P_h, hot_fluid)
    Q_max_hot = m_h * (h_hi - h_ho_min)
    h_co_max_hot = h_ci + Q_max_hot / m_c

    # Optional hard floor on the hot-stream outlet T
    # (e.g., brine reinjection limit set by silica saturation; in a real
    #  geothermal plant you would never allow the brine to leave the HX
    #  train below this temperature.)
    if T_h_out_min_C is not None:
        T_h_floor_K = T_h_out_min_C + 273.15
        if T_h_floor_K < T_hi:
            try:
                h_h_floor = PropsSI("H", "T", T_h_floor_K, "P", P_h, hot_fluid)
                Q_max_floor = m_h * (h_hi - h_h_floor)
                h_co_max_floor = h_ci + Q_max_floor / m_c
                # take the tightest of the three caps
                h_co_max_hot = min(h_co_max_hot, h_co_max_floor)
            except Exception:
                pass

    # The binding cap is the smaller one
    h_co_max = min(h_co_max_cold, h_co_max_hot)
    h_lo = h_ci + max(50.0, (h_co_max - h_ci) * 1e-4)

    common = dict(h_ci=h_ci, h_hi=h_hi, P_h=P_h, P_c=P_c,
                  m_h=m_h, m_c=m_c,
                  hot_fluid=hot_fluid, cold_fluid=cold_fluid,
                  U=U_W_m2K, n_seg=n_seg)

    _Q_max, _, _, _, A_max, dT_max = _tq_profile(h_co_max, **common)

    # 1) pinch constraint
    if dT_max >= pinch_K - 1e-6:
        h_pinch = h_co_max
    else:
        h_pinch = brentq(lambda h: _tq_profile(h, **common)[5] - pinch_K,
                         h_lo, h_co_max, xtol=1.0, rtol=1e-6)

    # 2) area constraint
    if A_max <= A_m2:
        h_area = h_co_max
    else:
        h_area = brentq(lambda h: _tq_profile(h, **common)[4] - A_m2,
                        h_lo, h_co_max, xtol=1.0, rtol=1e-6)

    h_op = min(h_pinch, h_area)
    if abs(h_op - h_co_max) < 1.0:
        limiting = "thermo-cap"
    elif h_op == h_pinch:
        limiting = "pinch"
    else:
        limiting = "area"

    Q_op, T_h, T_c, Q_arr, A_req, min_dT = _tq_profile(h_op, **common)

    h_h_out = h_hi - Q_op / m_h
    h_c_out = h_op

    try:
        x_c = PropsSI("Q", "H", h_c_out, "P", P_c, cold_fluid)
        if not (0.0 <= x_c <= 1.0):
            x_c = None
    except Exception:
        x_c = None
    try:
        x_h = PropsSI("Q", "H", h_h_out, "P", P_h, hot_fluid)
        if not (0.0 <= x_h <= 1.0):
            x_h = None
    except Exception:
        x_h = None

    return dict(
        name=name, Q=Q_op, A_required=A_req, A_installed=A_m2,
        T_h_in_C=T_h_in_C, T_h_out_C=T_h[0] - 273.15,
        T_c_in_C=T_c_in_C, T_c_out_C=T_c[-1] - 273.15,
        P_h_bar=P_h_bar, P_c_bar=P_c_bar,
        m_h=m_h, m_c=m_c,
        h_h_in=h_hi, h_h_out=h_h_out,
        h_c_in=h_ci, h_c_out=h_c_out,
        x_c_out=x_c, x_h_out=x_h,
        pinch_K_achieved=min_dT, pinch_K_target=pinch_K,
        limiting=limiting,
        U_W_m2K=U_W_m2K,
        tq_T_h=T_h - 273.15, tq_T_c=T_c - 273.15, tq_Q=Q_arr,
    )
