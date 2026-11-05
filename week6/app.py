# app.py
# LLM-CAPP Streamlit UI — Week 6
# Features:
#   - Analysis Mode (Route Only / Full Analysis)
#   - Machine Preference dropdown (Auto / Prefer Lathe / Prefer Milling)
#   - Color-coded route flow (Lathe=Yellow, Milling=Blue, Shared=Purple)
#   - INR costs (1 USD = Rs.96.095)
#   - Dimension-based time calculation with multi-pass depth
#   - Time breakdown: Machining + Tool Change + Position Change + Changeover
#   - Cost breakdown with bar chart
#   - Per-operation detail table

import streamlit as st
import sys, os, time, json
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load .env from week6 folder
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week1")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week2")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week3")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week4")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week5")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))     # repo root: the capp package (Steps 10-12)

from tokenizer import tokenize
from material_tokens import MATERIALS
from feature_vocab import GEOMETRY_FEATURES, FEATURE_TO_OPERATIONS, get_machine_type
from agents import (time_agent, cost_agent, energy_agent, efficiency_agent,
                    time_breakdown, TOOL_CHANGE_TIME_MIN,
                    POSITION_CHANGE_TIME_MIN, CHANGEOVER_TIME_MIN,
                    SAME_TOOL_GROUPS, USD_TO_INR)
from nsga2 import run_nsga2
from fsm_validator import validate_sequence
from self_corrector import self_correct
from precedence_graph import PRECEDENCE_EDGES
from route_builder import (generate_valid_routes, is_complete,
                           LATHE_ONLY_OPS, MILLING_ONLY_OPS)
from capp import analysis_view as AV      # physics + feasibility numbers (analysis only; no Streamlit inside)
from capp import process_plan as PP       # route sheet (Step 19): machine, tool, parameters, time per operation
from capp import routesheet as RS         # ... and its PDF / XLSX / HTML
from capp import dimensions as DIM        # hole diameters read from the drawing (Step 14): parse, validate, pre-fill rule
from capp import op_dims as OPD           # the dimensions of every single operation, filled from the drawing (Step 51)
from capp import calculator as CALC       # time and cost the way the Time-and-Cost-calculator does it (Step 52)
from capp import home_stats as HS         # the numbers and texts of the home page, read from the code (Step 52)
from capp import extraction_review as ER  # evidence, selective confirmation and the corrections log of the extraction (Step 20)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="LLM-CAPP | Intelligent Process Planning",
    page_icon="⚙️", layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.hero-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
.hero-sub   { font-size: 0.9rem; color: #6b7280; margin-top: 2px; margin-bottom: 14px; }
.stat-card { background: linear-gradient(135deg, #f8fafc, #e2e8f0);
    border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; text-align: center; }
.stat-num  { font-size: 1.7rem; font-weight: 700; color: #0f3460; }
.stat-lbl  { font-size: 0.72rem; color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.5px; margin-top: 4px; }
.step-hdr { background: linear-gradient(90deg, #0f3460, #1a5276); color: white;
    padding: 9px 16px; border-radius: 10px; font-weight: 600; margin: 12px 0 8px 0; }
.route-box { background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #34d399; border-radius: 14px; padding: 16px;
    text-align: center; margin: 8px 0; line-height: 2.2; }
.lathe-op    { background:#fef3c7; color:#92400e; padding:5px 11px;
    border-radius:8px; font-size:0.82rem; font-weight:500;
    display:inline-block; margin:3px 2px; }
.mill-op     { background:#dbeafe; color:#1e40af; padding:5px 11px;
    border-radius:8px; font-size:0.82rem; font-weight:500;
    display:inline-block; margin:3px 2px; }
.shared-op   { background:#ede9fe; color:#5b21b6; padding:5px 11px;
    border-radius:8px; font-size:0.82rem; font-weight:500;
    display:inline-block; margin:3px 2px; }
.universal-op{ background:#0f3460; color:white; padding:5px 11px;
    border-radius:8px; font-size:0.82rem; font-weight:500;
    display:inline-block; margin:3px 2px; }
.changeover-chip { background:#fef3c7; color:#92400e; padding:5px 11px;
    border-radius:8px; font-size:0.8rem; font-weight:600;
    display:inline-block; margin:3px 2px; border:1px dashed #d97706; }
.arrow { color:#94a3b8; margin:0 3px; }

/* Red Extract Features button */
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] .stButton button {
    background-color: #e53935 !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] button:hover,
section[data-testid="stSidebar"] .stButton button:hover {
    background-color: #c62828 !important;
    color: white !important;
}
/* Keep primary (Generate) button blue */
section[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #0f3460 !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background-color: #1a5276 !important;
}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────
MODEL_CALC = "\U0001F9EE Calculator model (your Time-and-Cost-calculator)"
MODEL_V1 = "\U0001F4C8 v1 model (numbers of the report)"
BASE_TIMES = {
    "Facing": 5, "Center Drilling": 3, "Drilling": 8, "Reaming": 4,
    "Inspection": 5, "Boring": 10, "Chamfering": 3, "External Threading": 7,
    "Plain/Cylindrical Turning": 9, "Taper Turning": 10, "Step Turning": 9,
    "Grooving/Necking": 6, "Parting-off": 4, "Knurling": 3, "Forming": 6,
    "Internal Grooving": 7, "Tapping": 5, "Counterboring": 4,
    "Countersinking": 3, "Contour Turning": 12, "Undercutting": 4,
    "Eccentric Turning": 11, "Polishing/Burnishing": 6, "Face Milling": 6,
    "Slab/Peripheral Milling": 10, "Surface Contouring": 15, "Slot Milling": 7,
    "T-Slot Milling": 9, "Dovetail Milling": 8, "Woodruff Keyway Milling": 5,
    "Pocket Milling": 9, "Profile Milling": 8, "Spotfacing": 3,
    "Corner Rounding/Filleting": 4, "Gear/Spline Milling": 14,
    "Thread Milling": 8, "Angular Milling": 7, "Gang Milling": 6,
    "Form Milling": 9, "Helical Milling": 11, "Engraving": 6,
}


def op_class(op):
    if op in ("Facing", "Inspection"): return "universal"
    if op in LATHE_ONLY_OPS:           return "lathe"
    if op in MILLING_ONLY_OPS:         return "mill"
    if op == "--- Machine Changeover ---": return "changeover"
    return "shared"


def make_route_flow(steps):
    html = ""
    for i, s in enumerate(steps):
        if i > 0:
            html += '<span class="arrow">→</span>'
        cls = op_class(s)
        if cls == "changeover":
            html += '<span class="changeover-chip">⚙ Machine Changeover</span>'
        elif cls == "universal":
            html += f'<span class="universal-op">{s}</span>'
        elif cls == "lathe":
            html += f'<span class="lathe-op">{s}</span>'
        elif cls == "mill":
            html += f'<span class="mill-op">{s}</span>'
        else:
            html += f'<span class="shared-op">{s}</span>'
    return html


def get_route_label(steps):
    has_lathe   = any(s in LATHE_ONLY_OPS  for s in steps)
    has_milling = any(s in MILLING_ONLY_OPS for s in steps)
    if has_lathe and has_milling: return "Lathe + Milling (with changeover)"
    if has_lathe:   return "Lathe Only"
    if has_milling: return "Milling Only"
    return "Shared Ops"


def make_radar(t, c, e, eff, scores=None):
    """scores = (time, cost, energy) already on a 0-100 scale (calculator mode: relative to the best route); without them the v1 scales."""
    cats = ['Time', 'Cost', 'Energy', 'Efficiency', 'Time']
    if scores is None:
        scores = (max(0,100-t*1.2), max(0,100-c*0.005), max(0,100-e*15))
    vals = [scores[0], scores[1], scores[2], eff, scores[0]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=7, color='#3b82f6')))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100], showticklabels=False),
                   bgcolor='rgba(0,0,0,0)'),
        showlegend=False, margin=dict(l=50,r=50,t=25,b=25),
        height=280, paper_bgcolor='rgba(0,0,0,0)')
    return fig


def make_comparison_chart(agent_data):
    if len(agent_data) < 2: return None
    colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6']
    fig = go.Figure()
    for i, row in enumerate(agent_data):
        fig.add_trace(go.Bar(
            name=row["Route"],
            x=["Time (min)", "Cost (₹/100)", "Energy (kWh)"],
            y=[row["Time (min)"], row["Cost (₹)"]/100, row["Energy (kWh)"]],
            marker_color=colors[i % len(colors)]
        ))
    fig.update_layout(barmode='group',
        margin=dict(l=30,r=20,t=25,b=35), height=260,
        legend=dict(orientation="h", y=1.12),
        paper_bgcolor='rgba(0,0,0,0)')
    return fig


def make_time_bar(bd):
    fig = go.Figure(go.Bar(
        x=["Machining", "Tool Changes", "Position Changes", "Changeover"],
        y=[bd['machining_time'], bd['tool_change_time'],
           bd['position_change_time'], bd['changeover_time']],
        marker_color=["#3b82f6", "#f59e0b", "#8b5cf6", "#ef4444"],
        text=[f"{v} min" for v in [bd['machining_time'], bd['tool_change_time'],
              bd['position_change_time'], bd['changeover_time']]],
        textposition="outside"
    ))
    fig.update_layout(
        yaxis_title="Time (minutes)",
        margin=dict(l=40,r=20,t=30,b=30), height=300,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig


# ── Feasibility / physics panels (numbers come from capp/analysis_view.py) ──
STATUS_HELP = ("Source status of a rule: **web** = a public machining reference (see the sources table), **derived** = follows from the v1 "
               "operation vocabulary, **model** = comes from the cutting-force model, **assumption** = an assumed value that must be replaced "
               "by the department's data. No rule is yet checked against a handbook page.")


def _finding_text(v):
    txt = f"**{v['rule_id']}** `[{v['source_status']}]` — {v['message']}"
    if v.get("action"):
        txt += f"  \n↳ suggested action: `{v['action']}`"
    return txt


def show_feasibility(fv, key):
    """Hard findings in red, risks in amber, the rules that could not be evaluated listed with the reason."""
    if fv is None:
        return
    for v in fv["infeasible"]:
        st.error("❌ " + _finding_text(v))
    for v in fv["risks"]:
        st.warning("⚠️ " + _finding_text(v))
    for v in fv["part_level"]:
        st.warning("📐 **Part level** (not about the order of the operations) — " + _finding_text(v))
    if not (fv["infeasible"] or fv["risks"] or fv["part_level"]):
        st.success(f"✅ No finding from the {fv['evaluated']} of {fv['n_rules']} rules that could be evaluated for this route.")
    ne = fv["not_evaluated"]
    if ne:
        with st.expander(f"ℹ️ {len(ne)} rule(s) not evaluated — and why"):
            st.dataframe([{"Rule": r, "Why not": why} for r, why in ne], use_container_width=True, hide_index=True)
    for note in fv["notes"]:
        st.caption("ℹ️ " + (note if isinstance(note, str) else " — ".join(str(x) for x in note)))


def _num(x, fmt, dash="—"):
    return dash if x is None else format(x, fmt)


def show_physics(pv, batch, key, covered_by_rules=True):
    """Metric cards, per-operation table and plots, Ra panel, speed sensitivity.
    covered_by_rules: the feasibility section above already shows the power and Ra findings, so only the other flags are repeated here."""
    if pv is None:
        return
    c = pv["cards"]
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("⚡ Cutting energy (kWh)", f"{c['energy_kwh']:.3f}", delta="cut only, lower bound", delta_color="off",
                  help="Kienzle cutting force x removed volume. Idle power, spindle acceleration and coolant are not included.")
    with m2:
        st.metric("🧱 Mean kc (N/mm²)", _num(c['kc_mean_n_mm2'], '.0f'),
                  delta=f"{_num(c['specific_energy_j_mm3'], '.2f')} J/mm³", delta_color="off",
                  help="Specific cutting force of the material at the chip thickness of each operation (Kienzle).")
    with m3:
        st.metric("🔪 Tool wear (₹ / part)", f"{c['wear_cost_inr']:.2f}", delta=f"₹{c['wear_cost_batch_inr']:,.0f} per {batch} parts", delta_color="off",
                  help="Taylor tool life at the standard speed (assumed per material) x an assumed price per cutting edge.")
    with m4:
        st.metric("🔌 Peak spindle power (kW)", f"{c['peak_spindle_kw']:.2f}",
                  delta=f"{_num(None if c['max_utilisation'] is None else c['max_utilisation'] * 100, '.0f')}% of {c['machine_power_kw']:g} kW",
                  delta_color="off", help="Highest power any operation needs at the standard speed and feed. The machine size is assumed.")
    with m5:
        icon = AV.RISK_ICON[c["ra_risk"]]
        if c["ra_required_um"]:
            sub = f"best {c['ra_best_um']:.2g} µm vs {c['ra_required_um']:g} µm"
        else:
            sub = "no required Ra given"
        st.metric("✨ Ra risk", f"{icon} {AV.RISK_SHORT[c['ra_risk']]}", delta=sub, delta_color="off",
                  help="Optimistic: the finest process of the route is assumed to make the critical surface.")

    for fl in (pv["other_flags"] if covered_by_rules else pv["flags"]):
        st.warning("⚠️ " + fl)
    if c["ra_risk"] in ("medium", "high") and c["ra_required_um"]:
        txt = f"**Ra {c['ra_required_um']:g} µm required** — {AV.RISK_TEXT[c['ra_risk']]}. Best the route reaches: {c['ra_best_um']:.2g} µm ({c['ra_best_op']})."
        if c["max_turning_feed_mm_rev"]:
            txt += f" For turning to reach it the feed must not exceed about {c['max_turning_feed_mm_rev']:.3f} mm/rev."
        st.info(txt)

    st.markdown("##### Per-operation physics")
    rows = pv["ops"]
    st.dataframe(rows, use_container_width=True, hide_index=True, column_config={
        "Cutting time (min)": st.column_config.NumberColumn(format="%.2f"),
        "kc (N/mm²)": st.column_config.NumberColumn(format="%.0f"),
        "MRR (cm³/min)": st.column_config.NumberColumn(format="%.1f"),
        "Power (kW)": st.column_config.NumberColumn(format="%.2f"),
        "Spindle load (%)": st.column_config.NumberColumn(format="%.0f"),
        "Energy (kWh)": st.column_config.NumberColumn(format="%.4f"),
        "Tool wear (₹)": st.column_config.NumberColumn(format="%.2f"),
        "Ra (µm)": st.column_config.NumberColumn(format="%.2f"),
    })
    cols = st.columns(2)
    for col, field, title, color, suffix in ((cols[0], "Energy (kWh)", "Cutting energy per operation (kWh)", "#3b82f6", "energy"),
                                             (cols[1], "Tool wear (₹)", "Tool wear per operation (₹ / part)", "#f59e0b", "wear")):
        pts = [(r["Operation"], r[field]) for r in rows if r[field]]
        if not pts:
            continue
        fig = go.Figure(go.Bar(x=[p[0] for p in pts], y=[p[1] for p in pts], marker_color=color))
        fig.update_layout(title=title, margin=dict(l=40, r=10, t=40, b=80), height=300,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis_tickangle=-40)
        with col:
            st.plotly_chart(fig, use_container_width=True, key=f"{key}_{suffix}")

    st.markdown("##### What if the cutting speed changes?")
    st.dataframe([{"Cutting speed": f"×{s['vc_factor']:.1f}", "Cutting time (min)": s["cutting_time_min"],
                   "Cutting energy (kWh)": s["energy_kwh"], "Tool wear (₹ / part)": s["wear_cost_inr"],
                   "Peak spindle power (kW)": s["peak_spindle_kw"],
                   "Over machine power": ", ".join(s["ops_over_power"]) or "—"} for s in pv["sensitivity"]],
                 use_container_width=True, hide_index=True, column_config={
        "Cutting time (min)": st.column_config.NumberColumn(format="%.2f"),
        "Cutting energy (kWh)": st.column_config.NumberColumn(format="%.4f"),
        "Tool wear (₹ / part)": st.column_config.NumberColumn(format="%.2f"),
        "Peak spindle power (kW)": st.column_config.NumberColumn(format="%.2f")})
    st.caption("A faster cut is shorter and draws more power and wears the tool faster; the energy of the cut itself stays the same. "
               "Fixed table (not a slider) so that the results below do not reset.")


def _download(label, data, file_name, mime, key):
    """A download button that does not rerun the app (a rerun would clear the results: they exist only while 'Generate' is pressed)."""
    try:
        st.download_button(label, data, file_name=file_name, mime=mime, key=key, on_click="ignore")
    except TypeError:                                   # an older Streamlit: no on_click="ignore"
        st.download_button(label, data, file_name=file_name, mime=mime, key=key)


def show_route_sheet(steps, part, key, context):
    """The route sheet of the selected route (Step 19, capp/routesheet.py): the table on screen, the document as a download.
    The route is already chosen: the sheet describes it and never changes it."""
    st.markdown("---")
    st.markdown("#### 📄 Route Sheet (process plan)")
    try:
        plan = PP.build_plan(steps, part, context=context)
    except Exception as exc:                            # the sheet is an extra: it must not stop the page
        st.warning(f"⚠️ The route sheet could not be built: {type(exc).__name__}: {exc}")
        return
    st.warning("📝 " + RS.draft_banner(plan))
    rows = [{"No.": r["seq"], "Operation": r["operation"], "Machine": r["machine"], "Tool": r["tool"], "Vc m/min": r["vc"], "n rpm": r["rpm"],
             "Feed": r["feed"], "ap × passes": r["depth"], "Time min": r["time"], "Notes": r["notes"]}
            for r in RS.display_rows(plan) if r["kind"] == "op"]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    t = plan["totals"]
    st.caption(f"Estimated {t['total_min']:.1f} min per part (cutting {t['cutting_min']:.1f} + tool changes {t['tool_change_min']:.0f} + position changes "
               f"{t['position_change_min']:.0f} + changeovers {t['changeover_min']:.0f}). * = assumed. The numbers are the v1 model's; the ranges they are compared "
               "with are from web pages, not the handbook.")
    stamp = RS._now()
    name = f"route_sheet_{plan['part']['part_id'] or 'part'}"
    with st.expander("⬇️ Download the route sheet"):
        _download("📄 HTML (print to PDF)", RS.to_html(plan, stamp), name + ".html", "text/html", key + "_html")
        for ext, label, mime, fn in (("pdf", "📕 PDF", "application/pdf", RS.to_pdf_bytes),
                                     ("xlsx", "📗 Excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", RS.to_xlsx_bytes)):
            try:
                _download(label, fn(plan, stamp), f"{name}.{ext}", mime, f"{key}_{ext}")
            except RS.RouteSheetError as exc:
                st.caption(f"{label}: {exc}")


# ── Extraction review (Step 20, report 6.5): evidence, selective confirmation, corrections log ──
CORRECTIONS_LOG = os.environ.get("CAPP_CORRECTIONS_LOG") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                          "results", "corrections_log.jsonl")
SEEN_BY_TEXT = {"both": "both passes", ER.UNKNOWN: "-"}


def _decide(feature, decision):
    """Callback of the Accept / Reject buttons: runs before the page is drawn again, so the list and the Generate button already agree."""
    feats, dec = ER.decide(st.session_state.get("confirmed_features", []), st.session_state.get("feature_decisions", {}), feature, decision)
    st.session_state["confirmed_features"] = feats
    st.session_state["feature_decisions"] = dec


def _map_name(name, feature):
    feats, mp = ER.map_unrecognised(st.session_state.get("confirmed_features", []), st.session_state.get("unrec_mappings", {}), name, feature)
    st.session_state["confirmed_features"] = feats
    st.session_state["unrec_mappings"] = mp


def _seen_by_text(row):
    if row["agreement"] == "one":
        return (row["seen_by"][0] + " only") if row["seen_by"] else "one pass only"
    if row["agreement"] == "both" and row["seen_by"] and set(row["seen_by"]) != set(ER.PASSES):
        return " + ".join(row["seen_by"])                  # e.g. "geometry + checklist", or all three passes
    return SEEN_BY_TEXT.get(row["agreement"], "-")


def show_review_panel(extraction_info, review):
    """The drawing beside the feature table, with the evidence of each feature. Strict mode only: Accept / Reject for the highlighted ones."""
    decisions = st.session_state.get("feature_decisions", {})
    features = st.session_state.get("confirmed_features", [])
    st.markdown("#### 🔎 Review the detected features")
    left, right = st.columns([1, 1.4])
    with left:
        img = st.session_state.get("uploaded_image_bytes")
        if img:
            try:
                st.image(img, caption="Your drawing", use_container_width=True)
            except Exception:                                   # a drawing that cannot be shown must not hide the table
                st.caption("(the drawing could not be shown)")
    with right:
        strict = ER.confirm_mode()
        if review["has_agreement"] and strict:
            st.caption("Seen by **two or more passes** = pre-accepted. Seen by **one pass only** = highlighted: accept or reject "
                       "it before you can generate the route. Add what the AI missed, or remove a pre-accepted feature, with the Features list in "
                       "the sidebar: those edits go to the corrections log.")
        elif review["has_agreement"]:
            st.caption("**Every feature the AI found is already in your list.** *Seen by* shows which passes found it (geometry, annotation, "
                       "checklist): a feature that only one pass found has a lower confidence, so check it against the drawing. Remove a wrong "
                       "feature, or add one the AI missed, with the Features list in the sidebar: those edits go to the corrections log.")
        else:
            st.caption("This extraction has no per-pass data, so every feature is included. Edit the Features list in the sidebar if needed.")
        for n in review["notes"]:
            st.warning("⚠️ " + n)
        label = ER.status(review, decisions, features)
        st.dataframe([{"Feature": r["feature"], "Seen by": _seen_by_text(r), "Confidence": r["confidence"] or "-", "Status": label[r["feature"]],
                       "Evidence": r["evidence_text"] or "-"} for r in review["rows"]], use_container_width=True, hide_index=True)
        if not any(r["evidence_text"] for r in review["rows"]):
            notes = ER.pass_notes(extraction_info)
            if notes:
                st.caption("Notes of the passes: " + "; ".join(f"{n}: {t}" for n, t in notes.items()))
        for r in review["rows"]:
            if not (r["needs_confirmation"] and strict):
                continue
            f = r["feature"]
            c0, c1, c2 = st.columns([3, 1, 1])
            c0.markdown(f"**{f}** — {_seen_by_text(r)}" + (f": {r['evidence_text']}" if r["evidence_text"] else ""))
            c1.button("✅ Accept", key=f"acc_{f}", on_click=_decide, args=(f, ER.ACCEPT), disabled=label[f] == "accepted")
            c2.button("❌ Reject", key=f"rej_{f}", on_click=_decide, args=(f, ER.REJECT), disabled=label[f] == "rejected")
        mappings = st.session_state.get("unrec_mappings", {})
        for u in review["unrecognised"]:
            seen = ", ".join(u["seen_by"]) or "the extraction"
            if u["name"] in mappings:
                st.caption(f"✓ “{u['name']}” (not in the vocabulary) was mapped to {mappings[u['name']]}.")
            elif u["suggestion"]:
                c0, c1 = st.columns([3, 2])
                c0.caption(f"⚠️ “{u['name']}” ({seen}) is not in the vocabulary. Nearest feature: **{u['suggestion']}**.")
                c1.button(f"Use {u['suggestion']}", key=f"map_{u['name']}", on_click=_map_name, args=(u["name"], u["suggestion"]))
            else:
                st.caption(f"⚠️ “{u['name']}” ({seen}) is not in the vocabulary and has no close match: add the right feature in the sidebar, or ignore it.")
        show_dimensions_table(extraction_info)


def show_dimensions_table(extraction_info):
    """Step 50: the dimensions the AI read from the drawing, beside the feature table. Only what is written; the sidebar takes the whole-part
    values from here, and anything not written keeps its default (said under Part Dimensions)."""
    pdm = extraction_info.get("part_dimensions")
    if not pdm:
        return
    st.markdown("**📐 Dimensions read from the drawing**")
    read = pdm.get("read")
    dim_report = extraction_info.get("dimensions") or {}
    if not read:
        st.caption("The dimensions could not be read (" + str(dim_report.get("error") or "unknown reason") + "). Type them in Part Dimensions.")
        return
    rows = []
    ov = read.get("overall") or {}
    if ov:
        nums = ", ".join((f"\u00d8{v:g}" if k == "diameter" else f"{k} {v:g}") for k, v in ov.items())
        rows.append({"Item": "Whole part", "As written": "", "Numbers (mm unless stated)": nums})
    for f in read.get("features") or []:
        rows.append({"Item": f["feature"], "As written": f.get("text", ""), "Numbers (mm unless stated)": DIM.feature_numbers_text(f) or "-"})
    if read.get("ra_um"):
        rows.append({"Item": "Surface finish", "As written": "", "Numbers (mm unless stated)": "Ra " + ", ".join(f"{v:g}" for v in read["ra_um"]) + " um"})
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Only numbers that are written on the drawing. Whole-part values go to Part Dimensions in the sidebar (Full Analysis mode); "
                   "the numbers of a feature go to the operation that makes it, in *Operations and their dimensions* below the table.")
    else:
        st.caption("No dimension is written on the drawing, or none the AI could read: Part Dimensions keep their defaults.")
    if read.get("dropped"):
        st.caption("Not used (unreadable, or outside the range of the app): " + "; ".join(read["dropped"][:8]))
    if read.get("note"):
        st.caption("AI note: " + read["note"])


def _calc_step(field):
    return 0.01 if field in ("feed", "depth of cut", "pitch", "feed per tooth") else 1.0


def show_op_dimensions(features, dims, hole_diameter, expanded=True, calc_out=None):
    """Step 51: every operation in a dropdown, the ones the features need already selected, each with the dimensions of its own cut, filled
    from the drawing (the whole-part value where the drawing gives none) and editable. Returns dims["per_op"] ({} when nothing is read or
    edited, which leaves every result as it was). Drawn in every run, in both branches of the page, so the number boxes keep their values.

    Step 52, calculator model (calc_out is a dict): an operation that has a process in the Time-and-Cost-calculator shows the inputs of THAT process
    (feed, rpm, approach, ...), filled from the drawing and the part where they are the same thing and with the calculator's defaults otherwise; the
    result is calc_out["edits"] (inputs the user changed) and calc_out["own"] (numbers of the drawing per operation). The other operations keep the
    four boxes above and their numbers are returned as per_op."""
    extraction = None if manual_input else st.session_state.get("extraction_result")
    calc_mode = calc_out is not None
    calc_options = list(OPD.ALL_OPS) + list(CALC.CALCULATOR_ONLY)
    with st.expander("\U0001F6E0\uFE0F Operations and their dimensions", expanded=expanded):
        if calc_mode:
            st.caption(f"Pick the operations (all {len(calc_options)} are in the list: the 41 of the planner and the {len(CALC.CALCULATOR_ONLY)} items of your calculator that no "
                       "operation maps to; the ones your features need are selected for you). **An operation you add here is added to the job's time and cost, one you take out "
                       "leaves them.** An operation that has a process in "
                       "your Time-and-Cost-calculator (or a milling process, from the formulas of the sheets) shows that process's inputs: numbers written on the "
                       "drawing for the feature it makes (\U0001F4D0) and the Part Dimensions fill what they mean, every other input starts at the calculator's "
                       "default (milling: the cutter and cutting data of the v1 model for the material). Change any box; time and cost follow. This does not change the route the planner makes (only the job that is costed). An operation of the route that is not in the list here uses the "
                       "same starting values. Each operation can be in the job once.")
        else:
            st.caption("Pick the operations (all 41 are in the list; the ones your features need are selected for you). Each selected operation shows "
                       "only the dimensions the time model reads for it. Numbers written on the drawing for the feature it makes are filled in "
                       "(\U0001F4D0); the rest is the Part Dimensions value. You can change any box. This does not change the route the planner "
                       "makes: it sets the dimensions the time, cost and cutting figures use for each operation. An operation of the route that is "
                       "not selected here uses the Part Dimensions.")
        if calc_mode:
            OPD.sync_selection(st.session_state, features, CALC.SELECTION_KEY, CALC.SIGNATURE_KEY)
            chosen = st.multiselect("Operations", calc_options, key=CALC.SELECTION_KEY,
                                    help="The selection follows the features: it is set again when the features change. Add an operation that is missing, or take one out.")
            ops = [op for op in calc_options if op in chosen]
        else:
            OPD.sync_selection(st.session_state, features)
            chosen = st.multiselect("Operations", OPD.ALL_OPS, key=OPD.SELECTION_KEY,
                                    help="The selection follows the features: it is set again when the features change.")
            ops = [op for op in OPD.ALL_OPS if op in chosen]
        values = {}
        if calc_mode:
            cdims = {**dims, "hole_diameter": hole_diameter}
            drawn = OPD.drawing_values(extraction, features)
            own = {op: {d: c["value"] for d, c in row.items()} for op, row in drawn.items()}
            calc_ops = [op for op in ops if CALC.has_process(op)]
            plain_ops = [op for op in ops if not CALC.has_process(op)]
            autos, srcs = {}, {}
            for op in calc_ops:
                autos[op], srcs[op] = CALC.inputs_src(op, cdims, own.get(op), material)
            CALC.sync_state(st.session_state, autos)
        else:
            calc_ops, plain_ops, autos, srcs, own = [], ops, {}, {}, {}
        cells = OPD.cells(plain_ops, features, extraction, dims, hole_diameter)
        OPD.sync_state(st.session_state, cells)
        if not ops:
            st.caption("No operation selected: every operation uses the Part Dimensions" + (" and the calculator's starting values." if calc_mode else "."))
            if calc_mode:
                calc_out.update({"edits": {}, "own": own, "selected": [], "auto": OPD.autoselect(features)})
            return {}

        if calc_ops:
            st.markdown("**Operations with a process in the calculator model** (lathe and drilling: your Time-and-Cost-calculator; milling: the formulas of the sheets)")
        for op in calc_ops:
            with st.container(border=True):
                proc = CALC.process_of(op)
                st.markdown(f"**{op}** \u2192 " + ("milling formulas (sheets), process" if op in CALC.MILLING_OF else "calculator process") + f" *{proc}*")
                fields = CALC.fields_of(op)
                cols = st.columns(4)
                for n, (name, label, _default) in enumerate(fields):
                    with cols[n % 4]:
                        values[CALC.key_of(op, name)] = st.number_input(
                            label, min_value=0.0, step=_calc_step(name), format="%g", key=CALC.key_of(op, name))
                one = CALC.edits_from_values({op: autos[op]}, values)
                row = CALC.op_row(op, material, cdims, CALC.Settings(labor_per_hour, material_cost, one, own))
                line = CALC.describe_sources(op, srcs[op], one.get(op, {}))
                st.caption(f"\u23F1 {row['time_min']:.2f} min \u00b7 labour \u20b9{row['labor_cost']:,.2f} \u00b7 tool \u20b9{row['tool_cost'] + row['extra_cost']:,.0f}"
                           + (f"  \u2014  {row['note']}" if row["note"] else "") + (f"\n\n{line}" if line else ""))
        if plain_ops:
            if calc_mode:
                st.markdown("**Operations the calculator has no process for** (v1 cutting time with the dimensions below, + "
                            f"{CALC.FALLBACK_TOOL_SETTING_MIN:g} min tool setting and \u20b9{CALC.FALLBACK_TOOL_COST:g} tool cost, the calculator's defaults; "
                            "Inspection keeps its fixed v1 time)")
            widths = [2.3, 1, 1, 1, 1, 3]
            head = st.columns(widths)
            for col, text in zip(head, ["Operation"] + [OPD.LABEL[d] + " (mm)" for d in OPD.DIM_KEYS] + ["Where the numbers come from"]):
                col.markdown(f"**{text}**")
            for op in plain_ops:
                line = st.columns(widths)
                line[0].markdown(op)
                for i, dim in enumerate(OPD.DIM_KEYS):
                    if dim in OPD.uses(op):
                        lo, hi = OPD.LIMITS[dim]
                        values[OPD.key_of(op, dim)] = line[i + 1].number_input(
                            f"{op} {OPD.LABEL[dim]} (mm)", min_value=lo, max_value=hi, step=1.0, format="%g",
                            key=OPD.key_of(op, dim), label_visibility="collapsed")
                    else:
                        line[i + 1].markdown("\u2013")
                line[5].caption(OPD.row_note(cells[op], values, op))
        per_op = OPD.per_op_from_values(cells, values)
        drawn_n = sum(1 for op in plain_ops for c in cells[op].values() if c["source"] == "drawing")
        if calc_mode:
            edits = CALC.edits_from_values(autos, values)
            calc_out.update({"edits": edits, "own": own, "selected": ops, "auto": OPD.autoselect(features)})
            n_in = sum(len(v) for v in edits.values())
            st.caption(f"{len(calc_ops)} selected operation(s) use the calculator's inputs ({n_in} input(s) changed by you); "
                       f"{len(per_op)} of the {len(plain_ops)} others use dimensions of their own"
                       + (f" ({drawn_n} number(s) read from the drawing)" if drawn_n else "") + ".")
        else:
            st.caption(f"{len(per_op)} of {len(ops)} selected operations use dimensions of their own"
                       + (f" ({drawn_n} number(s) read from the drawing)" if drawn_n else " (no number from a drawing)")
                       + "; the others use the Part Dimensions.")
        return per_op


def show_calculator_breakdown(res, settings, key):
    """Step 52: the Result page of the Time-and-Cost-calculator for the selected route: the summary and one row per operation."""
    st.markdown("---")
    st.markdown("#### \U0001F9EE Time & Cost \u2014 as in your Time-and-Cost-calculator")
    hours = res["time_min"] / 60.0
    m1, m2, m3 = st.columns(3)                                           # two rows of three: six in a row cut the numbers
    with m1: st.metric("\u23F1\uFE0F Total Time", f"{res['time_min']:.2f} min", delta=f"{hours:.2f} h", delta_color="off")
    with m2: st.metric("\U0001F477 Labour Cost", f"\u20b9{res['labor_cost']:,.2f}", delta=f"\u20b9{settings.labor_per_hour:g}/h", delta_color="off")
    with m3: st.metric("\U0001F9F1 Material Cost", f"\u20b9{res['material_cost']:,.2f}")
    m4, m5, m6 = st.columns(3)
    with m4: st.metric("\U0001F527 Tool Cost", f"\u20b9{res['tool_cost']:,.2f}")
    with m5: st.metric("\u2795 Extra Process Cost", f"\u20b9{res['extra_cost']:,.2f}")
    with m6: st.metric("\U0001F4B0 TOTAL COST", f"\u20b9{res['total_cost']:,.2f}")
    st.caption(f"{res['n_calculator']} of {res['n_ops']} operations were timed by a process of the calculator, {res['n_milling']} by the milling formulas of the "
               f"sheets (Vf = n \u00d7 fz \u00d7 Z, face milling (L + D)/Vf, ...), {res['n_v1']} by the v1 model (no process or formula for them). Energy: {res['energy_kwh']} kWh from the v1 energy model \u2014 the calculator has no energy formula. "
               "Batch size does not change the cost: the calculator prices one job.")
    rows = []
    for n, r in enumerate(res["rows"], 1):
        marker = r["op"] == CALC.MARKER
        rows.append({"No.": "\u2013" if marker else n, "Operation": "\u2699\uFE0F Machine Changeover" if marker else r["op"],
                     "Process": r["process"] or "\u2013", "Time (min)": round(r["time_min"], 2), "Labour (\u20b9)": round(r["labor_cost"], 2),
                     "Tool (\u20b9)": round(r["tool_cost"], 2), "Extra (\u20b9)": round(r["extra_cost"], 2), "Source": r["source"],
                     "Inputs / note": ", ".join(f"{k} {v:g}" for k, v in r["fields"].items()) + (f" ({r['note']})" if r["note"] and r["fields"] else r["note"])})
    with st.expander("\U0001F4CB Per-Operation Time and Cost (inputs of the calculator)", expanded=True):
        st.dataframe(rows, use_container_width=True, hide_index=True)
    labels = [f"{n}. {r['op']}" if r["op"] != CALC.MARKER else "Changeover" for n, r in enumerate(res["rows"], 1)]
    fig_t = go.Figure(go.Bar(x=labels, y=[r["time_min"] for r in res["rows"]], marker_color="#3b82f6",
                             text=[f"{r['time_min']:.1f}" for r in res["rows"]], textposition="outside"))
    fig_t.update_layout(yaxis_title="Time (minutes)", margin=dict(l=40, r=20, t=30, b=30), height=300,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig_t, use_container_width=True, key=key + "_time")
    parts = [("Material", res["material_cost"]), ("Labour", res["labor_cost"]), ("Tool", res["tool_cost"]), ("Extra process", res["extra_cost"])]
    fig_c = go.Figure(go.Bar(x=[p for p, _ in parts], y=[v for _, v in parts], marker_color=["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6"],
                             text=[f"\u20b9{v:,.0f}" for _, v in parts], textposition="outside"))
    fig_c.update_layout(title=f"Cost Distribution \u2014 Total: \u20b9{res['total_cost']:,.2f} for the job", yaxis_title="Cost (\u20b9 INR)",
                        margin=dict(l=40, r=20, t=40, b=30), height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig_c, use_container_width=True, key=key + "_cost")


def change_lines(steps):
    """Step 56: where the tool changes and the position changes of a route happen (v1 model), one line each. The same rule as the Time Breakdown of the
    v1 model and of Route Only."""
    lines = []
    prev = None
    tc_n = pc_n = 0
    for i, step in enumerate(steps):
        if i == 0: prev = step; continue
        if step in ("Inspection", "--- Machine Changeover ---"): prev = step; continue
        if prev == "--- Machine Changeover ---": prev = step; continue
        same = any(prev in g and step in g for g in SAME_TOOL_GROUPS)
        if not same:
            tc_n += 1
            lines.append(f"🔩 **Tool Change #{tc_n}:** `{prev}` → `{step}` (+{TOOL_CHANGE_TIME_MIN} min)")
        if prev not in ("Facing",):
            pc_n += 1
            lines.append(f"📐 **Position Change #{pc_n}:** After `{prev}`, re-clamped before `{step}` (+{POSITION_CHANGE_TIME_MIN} min)")
        prev = step
    return lines


def show_v1_split(steps, material, dims, key):
    """Step 56, calculator model: what Route Only shows and the calculator's table does not: the machining / tool change / position change / changeover
    split of the route, where the changes happen, and the chart. These are the minutes of the v1 model (the calculator has its own tool setting time per
    operation), so the section says so and is not called a Time Breakdown: the total above is the calculator's."""
    st.markdown("---")
    st.markdown("#### \U0001F501 Tool, Position and Machine Changes of the Route (v1 model split)")
    bd = time_breakdown(steps, material, dims)
    total = time_agent(steps, material, dims)
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1: st.metric("\U0001F527 Machining", f"{bd['machining_time']} min")
    with b2: st.metric("\U0001F529 Tool Change", f"{bd['tool_change_time']} min",
                       delta=f"{bd['tool_changes']} × {TOOL_CHANGE_TIME_MIN}min", delta_color="off")
    with b3: st.metric("\U0001F4D0 Position", f"{bd['position_change_time']} min",
                       delta=f"{bd['position_changes']} × {POSITION_CHANGE_TIME_MIN}min", delta_color="off")
    with b4: st.metric("⚙️ Changeover", f"{bd['changeover_time']} min",
                       delta=f"{bd['changeovers']} × {CHANGEOVER_TIME_MIN}min", delta_color="off")
    with b5: st.metric("\U0001F4CA TOTAL (v1 model)", f"{total} min", delta=f"Material factor: {bd['material_factor']}×", delta_color="off")
    st.caption("The same split Route Only shows, with your dimensions: the v1 model's minutes for the planner's route. The time and cost above are the "
               "calculator's (its own tool setting time and formulas), so this total is a different model's figure: read it for where the changes "
               "happen, not as the job's time.")
    lines = change_lines(steps)
    if lines:
        with st.expander(f"ℹ️ Where changes occurred ({bd['tool_changes']} tool + {bd['position_changes']} position)", expanded=True):
            for line in lines:
                st.markdown(f"- {line}")
    st.plotly_chart(make_time_bar(bd), use_container_width=True, key=key)


def log_corrections(extraction_info, features):
    """One line of the corrections log per Generate (report 6.5), unless the same drawing was just logged in the same state (pressing
    Generate again after changing only the material must not count the drawing twice). The log is for analysis; a problem writing it
    never stops the page."""
    try:
        entry = ER.build_log_entry(extraction_info, features, st.session_state.get("feature_decisions"), st.session_state.get("unrec_mappings"),
                                   drawing_id=st.session_state.get("uploaded_name", ""), image_bytes=st.session_state.get("uploaded_image_bytes"))
        state = json.dumps([entry["image_sha256"], entry["drawing_id"], entry["model_output"], entry["final_features"], entry["events"]],
                           sort_keys=True)
        if st.session_state.get("_last_logged") == state:
            return
        ER.append_log(CORRECTIONS_LOG, entry)
        st.session_state["_last_logged"] = state          # only after the write worked: a failed write is tried again at the next Generate
    except Exception as exc:
        st.caption(f"(the corrections log could not be written: {type(exc).__name__}: {exc})")


def show_assumptions_and_sources(view):
    with st.expander("🧾 Assumptions behind these numbers"):
        for line in view["assumptions"]:
            st.markdown(f"- {line}")
    rules = AV.rules_table()
    with st.expander(f"📚 The {len(rules)} feasibility rules and their sources"):
        st.caption(STATUS_HELP)
        st.dataframe(rules, use_container_width=True, hide_index=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Part Configuration")
    st.markdown("---")

    material = st.selectbox("🧱 Material", list(MATERIALS))          # Step 55: 19 materials (the first seven are the old ones, in the old order)

    st.markdown("---")

    st.markdown("**🎯 Analysis Mode**")
    analysis_mode = st.radio(
        "What do you want?",
        ["🗺️ Route Only", "📊 Route + Full Analysis"],
        index=1
    )
    route_only = analysis_mode == "🗺️ Route Only"

    st.markdown("---")
    st.markdown("**🔩 Feature Input Method**")

    features = []
    extraction_info = None
    review = None
    pending_features = []

    INPUT_UPLOAD = "🖼️ Upload a drawing (AI extraction)"
    INPUT_MANUAL = "✍️ Select the features myself"
    input_method = st.radio("How do you give the features?", [INPUT_UPLOAD, INPUT_MANUAL], key="input_method",
                            help="Manual selection needs no image and no API key. The route, the physics and the feasibility "
                                 "check work the same way with either.")
    manual_input = input_method == INPUT_MANUAL

    if manual_input:
        features = st.multiselect("Features", GEOMETRY_FEATURES, key="manual_features",
                                  help="Pick every feature of the part. The hole diameter (Full Analysis mode) is typed below.")
    else:
        if "manual_features" in st.session_state:        # Streamlit drops the state of a hidden widget: keep the manual selection
            st.session_state["manual_features"] = st.session_state["manual_features"]
        uploaded_image = st.file_uploader("Upload a 2D engineering drawing / sketch",
            type=["png", "jpg", "jpeg"])

        use_evidence = st.checkbox("Ask the AI for a short evidence note per feature", key="use_evidence",
                                   help="A different prompt (more tokens, new API calls): every feature comes with the annotation text or the view it is "
                                        "based on. Off by default: the extraction is then exactly the one validated so far.")
        if uploaded_image is not None:
            if st.button("🔍 Extract Features from Image", use_container_width=True):
                with st.spinner("Analyzing image with AI vision model..."):
                    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week1")))
                    from image_feature_extractor import extract_features_from_image
                    img_bytes = uploaded_image.getvalue()
                    img_format = uploaded_image.type.split("/")[-1]
                    extraction_info = extract_features_from_image(img_bytes, image_format=img_format, with_dimensions=True,
                                                                  with_evidence=bool(use_evidence), with_checklist=True, with_part_dimensions=True)
                st.session_state["extraction_result"] = extraction_info
                st.session_state["uploaded_image_bytes"] = img_bytes
                st.session_state["uploaded_name"] = uploaded_image.name
                st.session_state["feature_decisions"] = {}
                st.session_state["unrec_mappings"] = {}
                # Step 49: every feature the extraction found goes into the list at once (strict mode, CAPP_CONFIRM_FLAGGED=1: only what
                # two passes saw; the rest waits for Accept / Reject)
                st.session_state["confirmed_features"] = (ER.initial_features(ER.build_review(extraction_info))
                                                          if extraction_info["success"] else extraction_info["features"])

        # Show extraction result (persisted in session_state across reruns)
        if "extraction_result" in st.session_state:
            extraction_info = st.session_state["extraction_result"]
            if extraction_info["success"]:
                conf = extraction_info["confidence"]
                conf_icon = "🟢" if conf == "high" else "🟡" if conf == "medium" else "🔴"
                st.success(f"{conf_icon} Detected (confidence: {conf}): {', '.join(extraction_info['features']) or 'none'}")
                if extraction_info["notes"]:
                    st.caption(f"AI notes: {extraction_info['notes']}")
                if extraction_info["rejected_features"]:
                    st.warning(f"⚠️ Not in the feature vocabulary (not used unless you map them in the review panel): "
                               f"{', '.join(extraction_info['rejected_features'])}")
                if "dimensions" in extraction_info:
                    holes = extraction_info.get("hole_diameters") or []
                    if holes:
                        st.caption("🕳️ Hole diameter(s) the AI read from the drawing: " + ", ".join(f"Ø{h:g}" for h in holes)
                                   + f". Ø{DIM.choose_hole_diameter(holes):g} (the largest) is pre-filled in Full Analysis mode: "
                                     "check it against the drawing.")
                    elif extraction_info["dimensions"].get("success"):
                        st.caption("🕳️ No hole diameter is written on the drawing (or none the AI could read): type it in Full Analysis mode if you know it.")
                    else:
                        st.caption("🕳️ Hole diameter not read: " + str(extraction_info["dimensions"].get("error") or "unknown reason")
                                   + ". Type it in Full Analysis mode if you know it.")
                if not extraction_info["features"]:
                    with st.expander("🔍 Debug: What did the AI actually say? (nothing was detected)"):
                        st.code(extraction_info.get("raw_response", "(empty)"), language="text")

                st.markdown("**✏️ Confirm or edit detected features:**")
                features = st.multiselect("Features (edit if AI missed/misread something)",
                    GEOMETRY_FEATURES, default=extraction_info["features"], key="confirmed_features")
                # session_state se sync karo — rerun pe multiselect reset ho jaata hai
                if not features and "confirmed_features" in st.session_state:
                    features = st.session_state["confirmed_features"]
                review = ER.build_review(extraction_info)
                pending_features = ER.pending(review, st.session_state.get("feature_decisions", {}), features)
                if pending_features:
                    st.warning(f"⚠️ {len(pending_features)} feature(s) seen by one pass only need your Accept or Reject before you can generate: "
                               + ", ".join(pending_features) + ". The buttons are in the main area, beside the drawing.")
            else:
                st.error(f"❌ Extraction failed: {extraction_info.get('error', 'unknown error')}")
                st.info("Try uploading a clearer image.")

    if features:
        l = [f for f in features if get_machine_type(f)=="Lathe"]
        m = [f for f in features if get_machine_type(f)=="Milling"]
        b = [f for f in features if get_machine_type(f)=="Both"]
        parts = []
        if l: parts.append(f"🟡 Lathe: {', '.join(l)}")
        if m: parts.append(f"🔵 Mill: {', '.join(m)}")
        if b: parts.append(f"🟣 Both: {', '.join(b)}")
        st.caption(" | ".join(parts))

    tolerance = st.selectbox("📏 Tolerance",
        ["0.005mm","0.01mm","0.02mm","0.05mm","0.1mm","0.5mm"], index=2)

    batch_size = st.slider("📦 Batch Size", 1, 10000, 500, step=10)

    st.markdown("---")
    if not route_only:
        st.markdown("**📐 Part Dimensions**")
        # Step 50: what is written on the drawing goes into the fields (once per detection: a later edit by the user is kept); what is not
        # written stays at the default, and the caption under the fields says which is which.
        part_keys = {"diameter": "dim_diameter", "length": "dim_length", "depth": "dim_depth", "width": "dim_width", "ra_um": "dim_ra"}
        values, st.session_state["part_prefill_sig"] = DIM.prefill_part(
            st.session_state.get("extraction_result"), st.session_state.get("part_prefill_sig"),
            {k: key in st.session_state for k, key in part_keys.items()})
        for k, v in values.items():
            st.session_state[part_keys[k]] = float(v)
        for k, v in {"dim_diameter": 25.0, "dim_length": 50.0, "dim_depth": 20.0, "dim_width": 20.0, "dim_ra": None}.items():
            st.session_state.setdefault(k, v)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            part_diameter = st.number_input("Diameter (mm)", min_value=1.0, max_value=500.0, step=1.0, key="dim_diameter")
            part_depth = st.number_input("Depth (mm)", min_value=1.0, max_value=300.0, step=1.0, key="dim_depth")
        with col_d2:
            part_length = st.number_input("Length (mm)", min_value=1.0, max_value=1000.0, step=1.0, key="dim_length")
            part_width = st.number_input("Width (mm)", min_value=1.0, max_value=500.0, step=1.0, key="dim_width")
        dims = {"diameter": part_diameter, "length": part_length,
                "depth": part_depth, "width": part_width}
        # Used only by the physics / feasibility panels (Step 12); the v1 agents never see these two.
        ra_choice = st.selectbox("✨ Required surface finish Ra (µm)", AV.RA_CHOICES, key="dim_ra",
                                 format_func=lambda v: "not specified" if v is None else f"{v:g}",
                                 help="Leave 'not specified' if the drawing gives no Ra; the Ra checks are then skipped.")
        # A diameter the AI read from the drawing is put in the field once per detection (a later edit by the user is kept).
        prefill_value, prefill_sig = DIM.prefill(st.session_state.get("extraction_result"), st.session_state.get("hole_prefill_sig"),
                                                 present="hole_dia_in" in st.session_state)
        if prefill_value is not None:
            st.session_state["hole_dia_in"] = float(prefill_value)
            st.session_state["hole_prefill_sig"] = prefill_sig
        hole_dia_in = st.number_input("🕳️ Hole diameter (mm, 0 = unknown)", min_value=0.0, max_value=200.0, step=1.0, key="hole_dia_in",
                                      help="Optional. Without it the deep-hole check cannot run and hole operations use the part diameter "
                                           "in the physics figures. Pre-filled when the AI read one from the drawing: check it.")
        required_ra = ra_choice
        hole_diameter = hole_dia_in or None
        _ex = st.session_state.get("extraction_result") or {}
        if _ex.get("success") and not manual_input and "part_dimensions" in _ex:
            _derived = (_ex["part_dimensions"] or {}).get("derived")
            if _derived:
                _written, _missing = DIM.describe_derived(_derived)
                if _written:
                    st.caption("📐 Read from your drawing: " + ", ".join(_written) + ". Check them against the drawing.")
                if _missing:
                    st.caption("Not written on the drawing, so the default stays (set it yourself): " + ", ".join(_missing) + ".")
                for _n in _derived.get("notes") or []:
                    st.caption("ℹ️ " + _n)
            else:
                st.caption("📐 The dimensions could not be read from the drawing ("
                           + str((_ex.get("dimensions") or {}).get("error") or "unknown reason") + "): the defaults stay, set them yourself.")
        st.markdown("---")
        st.markdown("**\U0001F9EE Time & Cost Model**")
        model_label = st.radio("Time and cost are calculated with", [MODEL_CALC, MODEL_V1], key="time_model",
                               help="Calculator: the formulas of your Time-and-Cost-calculator (feed, rpm, passes, setting times, labour per hour, "
                                    "material cost). v1: the model of the experiments and of the report.")
        calc_mode = model_label == MODEL_CALC
        labor_per_hour = st.number_input("\U0001F477 Labour cost (\u20b9/hour)", min_value=0.0, value=CALC.LABOR_PER_HOUR, step=50.0, key="calc_labor",
                                         disabled=not calc_mode)
        material_cost = st.number_input("\U0001F9F1 Material cost (\u20b9 per job)", min_value=0.0, value=CALC.MATERIAL_COST, step=100.0, key="calc_material",
                                        disabled=not calc_mode)
        st.caption("Calculator mode: lathe and drilling from your calculator, milling from the sheets' formulas; energy and the operations without any "
                   "formula stay on the v1 model (said in the results)."
                   if calc_mode else "v1 model: the numbers of the report.")
        st.markdown("---")
    else:
        dims = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}
        required_ra = None
        hole_diameter = None
        calc_mode = False
        labor_per_hour, material_cost = CALC.LABOR_PER_HOUR, CALC.MATERIAL_COST

    st.markdown("**🏭 Machine Preference**")
    pref_opts = {
        "🤖 Auto (Minimize Changeovers)": "auto",
        "🟡 Prefer Lathe":               "prefer_lathe",
        "🔵 Prefer Milling":             "prefer_milling",
    }
    pref_label = st.selectbox("Choose machine strategy", list(pref_opts.keys()))
    pref_value = pref_opts[pref_label]
    st.caption("Auto = system minimizes changeovers")

    st.markdown("---")
    run_btn = st.button("🚀 Generate Process Plan", type="primary", use_container_width=True,
                        disabled=bool(pending_features) and not manual_input)
    if pending_features and not manual_input:
        st.caption("Generate stays off until every highlighted feature is accepted or rejected.")
    st.markdown("---")
    st.caption("LLM-CAPP v2.0 | IIT Kharagpur SRIC")
    st.caption("Prof. Sankha Deb | Summer 2026")


# ── Main Page ─────────────────────────────────────────────────
st.markdown('<p class="hero-title">⚙️ LLM-Based Intelligent Process Planning</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Dynamic Route Builder · Machine-Aware Grouping · NSGA-II · INR Costing</p>', unsafe_allow_html=True)

if not run_btn:
    if review is not None and not manual_input and st.session_state.get("extraction_result", {}).get("success"):
        show_review_panel(st.session_state["extraction_result"], review)
        st.markdown("---")
    if not route_only and features:
        show_op_dimensions(features, dims, hole_diameter, expanded=True, calc_out={} if calc_mode else None)       # Steps 51, 52
        st.markdown("---")
    cards = HS.cards()
    for col, (n, l, tip) in zip(st.columns(len(cards)), cards):
        with col:
            st.markdown(f'<div class="stat-card" title="{tip}"><div class="stat-num">{n}</div>'
                        f'<div class="stat-lbl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    blurbs = HS.blurbs()
    for row in (blurbs[:3], blurbs[3:]):
        for col, (title, text) in zip(st.columns(3), row):
            with col:
                st.markdown(f"#### {title}")
                st.markdown(text)
    st.caption("👈 Configure part in sidebar → click **Generate Process Plan**")

else:
    # Image upload mode mein features session_state se lo (not in manual mode: an empty manual selection must stay empty,
    # and must not fall back to the features of an earlier extraction)
    if not features and not manual_input and "confirmed_features" in st.session_state:
        features = st.session_state["confirmed_features"]

    if not features:
        st.error("❌ Please select at least one feature!")
        st.stop()

    if not manual_input and st.session_state.get("extraction_result", {}).get("success"):
        log_corrections(st.session_state["extraction_result"], features)

    if not route_only:
        # Step 51: the same panel as before the click (collapsed), so its boxes keep their values; per_op is {} unless the drawing gave
        # numbers or the user typed some, and then dims is what it was
        calc_out = {} if calc_mode else None
        per_op = show_op_dimensions(features, dims, hole_diameter, expanded=False, calc_out=calc_out)
        if per_op:
            dims = {**dims, "per_op": per_op}
    calc_settings = cdims = None
    job_selected = job_auto = None
    if not route_only and calc_mode:
        # Step 52: the part for the calculator model = the dimensions above + the hole diameter; its settings = labour, material, the inputs the user
        # changed and the numbers of the drawing
        cdims = {**dims, "hole_diameter": hole_diameter}
        calc_settings = CALC.Settings(labor_per_hour, material_cost, (calc_out or {}).get("edits"), (calc_out or {}).get("own"))
        job_selected, job_auto = (calc_out or {}).get("selected"), (calc_out or {}).get("auto")        # Step 54: what the dropdown holds now / what the features selected

    part = {"material": material, "features": features,
            "tolerance": tolerance, "batch_size": batch_size, "dims": dims}
    pipeline_start = time.time()

    # STEP 1
    st.markdown('<div class="step-hdr">STEP 1 — Feature Tokenization</div>', unsafe_allow_html=True)
    tok = tokenize(part)
    if tok["success"]:
        chips = " ".join([f"`{t}`" for t in tok["tokens"]])
        st.success(f"✅ Token sequence: {chips}")
    else:
        st.error(f"❌ Tokenization failed: {tok['errors']}")
        st.stop()

    # STEP 2
    st.markdown('<div class="step-hdr">STEP 2 — Machine-Aware Route Generation + NSGA-II Optimization</div>', unsafe_allow_html=True)
    pareto = run_nsga2(material, batch_size, features=features, machine_preference=pref_value, max_routes=30)
    st.success(f"✅ Found **{len(pareto)}** Pareto-optimal route(s) — Strategy: *{pref_label}*")

    # nsga2.py machine-preference override / incomplete-fallback status
    # compute karta tha lekin UI kabhi dikhata hi nahi tha -- isliye jab
    # "Prefer Lathe/Milling" ke saath koi valid route nahi bana paata tha,
    # user ko koi warning nahi milti thi, bas silently Auto pe switch ho
    # jaata (ya worse, ek incomplete 2-step route "valid" dikh jaata).
    gen_status = getattr(run_nsga2, "last_llm_status", None)
    if gen_status:
        if gen_status.get("machine_pref_overridden"):
            st.warning(f"⚠️ {gen_status['reason']}")
        elif gen_status.get("fallback_incomplete"):
            st.error(f"❌ {gen_status['reason']} Neeche dikhaya gaya route features cover NAHI karta — features/tolerance/batch adjust karke dobara try karo.")

    test_routes = generate_valid_routes(features, max_routes=1, machine_preference=pref_value)
    if test_routes and test_routes[0].get("warnings"):
        for w in test_routes[0]["warnings"]:
            st.warning(f"⚠️ {w}")

    # STEP 3
    st.markdown('<div class="step-hdr">STEP 3 — FSM Validation & Self-Correction</div>', unsafe_allow_html=True)
    valid_routes = []
    for ind in pareto:
        clean = [s for s in ind.steps if s != "--- Machine Changeover ---"]
        if validate_sequence(clean)["valid"]:
            st.markdown(f"✅ `{ind.route_name}` — Valid")
            valid_routes.append(ind)
        else:
            corr = self_correct(clean, features=features, verbose=False)
            if corr["success"]:
                st.warning(f"🔧 `{ind.route_name}` — Auto-corrected")
                ind.steps = corr["corrected"]
                valid_routes.append(ind)
            else:
                st.error(f"❌ `{ind.route_name}` — Could not fix")

    elapsed = time.time() - pipeline_start

    # ═══════════════════════════════════════════════════
    # ROUTE ONLY MODE
    # ═══════════════════════════════════════════════════
    if route_only:
        st.markdown("---")
        st.markdown("### 🏆 Optimal Process Route")
        # valid_routes[0] tha pehle -- pareto list mein jo bhi order mein
        # pehle aaya, wo "optimal" ban jaata tha, actual highest-efficiency
        # nahi. Ab genuinely best-scoring route select karte hain.
        best_ind = None
        best_eff_ro = -1
        for ind in valid_routes:
            eff_ro = efficiency_agent(time_agent(ind.steps, material),
                                       cost_agent(ind.steps, material, batch_size),
                                       energy_agent(ind.steps, material))
            if eff_ro > best_eff_ro:
                best_eff_ro = eff_ro
                best_ind = ind
        best_steps = best_ind.steps if best_ind else None
        best_route_name = best_ind.route_name if best_ind else ""

        if best_steps:
            origin_badge = ""
            if best_route_name.startswith("Route_LLM"):
                was_corrected = "corrected" in best_route_name
                origin_badge = (f" &nbsp;|&nbsp; 🤖 **AI-Suggested Route**"
                               f"{' (self-corrected)' if was_corrected else ' (validated as-is)'}")
            st.markdown(f"**Machine Type:** `{get_route_label(best_steps)}` &nbsp;|&nbsp; **Strategy:** {pref_label} &nbsp;|&nbsp; **Pipeline Time:** {elapsed:.2f}s{origin_badge}")
            st.markdown(f'<div class="route-box">{make_route_flow(best_steps)}</div>', unsafe_allow_html=True)
            st.caption("🟡 Yellow=Lathe | 🔵 Blue=Milling | 🟣 Purple=Shared | ⚫ Dark=Universal | ⚙=Changeover")

            clean = [s for s in best_steps if s != "--- Machine Changeover ---"]
            if is_complete(clean, features):
                st.success(f"✅ Completeness Verified — All {len(features)} feature(s) covered")

            # Feasibility check (Step 12). No dimensions in this mode, so only the rules that need none can run;
            # the others are listed as "not evaluated" with the reason.
            st.markdown("---")
            st.markdown("#### 🛡️ Feasibility Check")
            ro_view = AV.analyse_route_view(best_steps, AV.make_part(material, features, tolerance, batch_size, dims=None))
            for err in ro_view["errors"]:
                st.warning("⚠️ " + err)
            show_feasibility(ro_view["feasibility"], "ro_feas")
            st.caption("Without part dimensions only the rules that need none can run. Use **Route + Full Analysis** for the "
                       "dimension-dependent checks and the cutting-energy / tool-wear figures.")
            with st.expander(f"📚 The {len(AV.rules_table())} feasibility rules and their sources"):
                st.caption(STATUS_HELP)
                st.dataframe(AV.rules_table(), use_container_width=True, hide_index=True)

            ro_source = "AI-suggested route" + (" (self-corrected)" if "corrected" in best_route_name else "") if best_route_name.startswith("Route_LLM") \
                else "Route Builder / NSGA-II"
            show_route_sheet(best_steps, AV.make_part(material, features, tolerance, batch_size, dims=None), "ro_sheet", {"source": ro_source})

            # Rough time breakdown
            st.markdown("---")
            st.markdown("#### ⏱️ Time Breakdown (Reference Dimensions)")
            st.caption("Ø25mm, L=50mm, depth=20mm — switch to Full Analysis for exact dimensions.")

            rough_t   = time_agent(best_steps, material)
            rough_c   = cost_agent(best_steps, material, batch_size)
            rough_e   = energy_agent(best_steps, material)
            rough_eff = efficiency_agent(rough_t, rough_c, rough_e)
            rough_bd  = time_breakdown(best_steps, material)

            b1,b2,b3,b4,b5 = st.columns(5)
            with b1: st.metric("🔧 Machining", f"{rough_bd['machining_time']} min")
            with b2: st.metric("🔩 Tool Change", f"{rough_bd['tool_change_time']} min",
                        delta=f"{rough_bd['tool_changes']} × {TOOL_CHANGE_TIME_MIN}min", delta_color="off")
            with b3: st.metric("📐 Position", f"{rough_bd['position_change_time']} min",
                        delta=f"{rough_bd['position_changes']} × {POSITION_CHANGE_TIME_MIN}min", delta_color="off")
            with b4: st.metric("⚙️ Changeover", f"{rough_bd['changeover_time']} min",
                        delta=f"{rough_bd['changeovers']} × {CHANGEOVER_TIME_MIN}min", delta_color="off")
            with b5: st.metric("📊 TOTAL", f"{rough_t} min")

            st.caption(f"ℹ️ Material factor for **{material}**: {rough_bd['material_factor']}×. "
                       f"Same-tool-group ops do **not** incur tool change time.")

            if rough_bd['tool_changes'] > 0 or rough_bd['position_changes'] > 0:
                lines = []
                prev = None
                tc_n = pc_n = 0
                for i, step in enumerate(best_steps):
                    if i == 0: prev = step; continue
                    if step in ("Inspection", "--- Machine Changeover ---"): prev = step; continue
                    if prev == "--- Machine Changeover ---": prev = step; continue
                    same = any(prev in g and step in g for g in SAME_TOOL_GROUPS)
                    if not same:
                        tc_n += 1
                        lines.append(f"🔩 **Tool Change #{tc_n}:** `{prev}` → `{step}` (+{TOOL_CHANGE_TIME_MIN} min)")
                    if prev not in ("Facing",):
                        pc_n += 1
                        lines.append(f"📐 **Position Change #{pc_n}:** After `{prev}`, re-clamped before `{step}` (+{POSITION_CHANGE_TIME_MIN} min)")
                    prev = step
                if lines:
                    with st.expander(f"ℹ️ Where changes occurred ({rough_bd['tool_changes']} tool + {rough_bd['position_changes']} position)", expanded=True):
                        for line in lines:
                            st.markdown(f"- {line}")

            st.plotly_chart(make_time_bar(rough_bd), use_container_width=True, key="ro_time_bar")

            ec1,ec2,ec3 = st.columns(3)
            with ec1: st.metric("💰 Est. Cost",  f"₹{rough_c:,.0f}")
            with ec2: st.metric("⚡ Est. Energy", f"{rough_e} kWh")
            with ec3: st.metric("📊 Efficiency",  f"{rough_eff}/100")

            st.info("💡 Rough estimates. Use **Route + Full Analysis** for dimension-specific accuracy.")

            if len(valid_routes) > 1:
                with st.expander(f"📋 All {len(valid_routes)} valid routes"):
                    for i, ind in enumerate(valid_routes):
                        rt = time_agent(ind.steps, material)
                        st.markdown(f"**Route {i+1}:** {' → '.join(ind.steps)} &nbsp;|&nbsp; ~{rt} min")

            st.markdown("#### Feature → Operation Mapping")
            for feat in features:
                ops  = FEATURE_TO_OPERATIONS[feat]["alternatives"][0]
                m    = get_machine_type(feat)
                icon = "🟡" if m=="Lathe" else "🔵" if m=="Milling" else "🟣"
                st.markdown(f"{icon} **{feat}** ({m}) → {', '.join(ops)}")

            st.balloons()
        else:
            st.error("❌ No valid route found.")
        st.stop()

    # ═══════════════════════════════════════════════════
    # FULL ANALYSIS MODE
    # ═══════════════════════════════════════════════════
    st.markdown('<div class="step-hdr">STEP 4 — Multi-Agent Evaluation (Time · Cost · Energy · Efficiency)</div>', unsafe_allow_html=True)
    agent_data = []
    best = None
    best_eff = -1
    # The part as the physics / feasibility modules read it (dims + required Ra + optional hole diameter).
    part_av = AV.make_part(material, features, tolerance, batch_size, dims=dims,
                           required_ra_um=required_ra, hole_diameter=hole_diameter)

    evals = []
    for ind in valid_routes:
        if calc_mode:                       # Step 52: time and cost as the Time-and-Cost-calculator computes them; energy = v1 (no formula there)
            job = CALC.plan_job(ind.steps, job_selected, job_auto) if job_selected is not None else ind.steps       # Step 54: operations added / taken out in the dropdown
            res = CALC.evaluate(job, material, cdims, calc_settings)
            evals.append((ind, round(res["time_min"], 2), round(res["total_cost"], 2), res["energy_kwh"], res))
        else:
            evals.append((ind, time_agent(ind.steps, material, dims), cost_agent(ind.steps, material, batch_size, dims),
                          energy_agent(ind.steps, material, dims), None))
    if calc_mode:
        effs = CALC.relative_efficiency([(t, c, e) for _, t, c, e, _ in evals])
        rel = CALC.relative_scores([(t, c, e) for _, t, c, e, _ in evals])
    else:
        effs = [efficiency_agent(t, c, e) for _, t, c, e, _ in evals]
        rel = [None] * len(evals)

    for (ind, t, c, e, res), eff, scores in zip(evals, effs, rel):
        has_co = "--- Machine Changeover ---" in ind.steps

        agent_data.append({
            "Route": ind.route_name,
            "Machine Type": get_route_label(ind.steps),
            "Changeover": "Yes ⚙️" if has_co else "No",
            "Time (min)": t, "Cost (₹)": c,
            "Energy (kWh)": e, "Efficiency": eff,
            **AV.route_row(ind.steps, part_av),        # physics + feasibility columns; they do NOT enter the efficiency score
        })

        if eff > best_eff:
            best_eff = eff
            best = {"route": ind.route_name, "steps": ind.steps,
                    "t": t, "c": c, "e": e, "eff": eff, "changeover": has_co, "calc": res, "scores": scores}

    st.dataframe(agent_data, use_container_width=True, hide_index=True)
    st.caption("The last six columns (cutting energy, kc, tool wear, Ra risk, hard issues, risks) come from the physics and feasibility "
               "models of Step 5 below. They describe each route; the efficiency score and the selection use only the first columns.")
    if calc_mode:
        st.caption("\U0001F9EE Calculator model: time and cost are computed the way your Time-and-Cost-calculator does (labour "
                   f"\u20b9{labor_per_hour:g}/h, material \u20b9{material_cost:g}); energy is the v1 figure (the calculator has no energy formula). "
                   "Efficiency = the average of 100 \u00d7 best/this route for time, cost and energy among these routes (100 = best in all three): the "
                   "fixed scales of the v1 efficiency were made for times of a few minutes and would give 0 to every route here.")

    if len(evals) > 1:
        with st.expander(f"📋 All {len(evals)} valid routes"):
            for i, (ind_i, t_i, _c, _e, _r) in enumerate(evals):
                st.markdown(f"**Route {i+1}:** {' → '.join(ind_i.steps)} &nbsp;|&nbsp; ~{t_i} min")

    comp = make_comparison_chart(agent_data)
    if comp:
        st.caption("📊 Route Comparison (Cost ÷ 100 for scale)")
        st.plotly_chart(comp, use_container_width=True, key="fa_comp")

    st.markdown("---")
    st.markdown("### 🏆 Optimal Process Plan")

    if best:
        origin_badge = ""
        if best["route"].startswith("Route_LLM"):
            was_corrected = "corrected" in best["route"]
            origin_badge = (f" &nbsp;|&nbsp; 🤖 **AI-Suggested Route**"
                           f"{' (self-corrected by Route Builder)' if was_corrected else ' (validated as-is)'}")
        st.markdown(f"**Machine Type:** `{get_route_label(best['steps'])}` &nbsp;|&nbsp; **Strategy:** {pref_label} &nbsp;|&nbsp; **Pipeline Time:** {elapsed:.2f}s{origin_badge}")

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("⏱️ Total Time",   f"{best['t']} min")
        with c2: st.metric("💰 Total Cost",   f"₹{best['c']:,.2f}")
        with c3: st.metric("⚡ Energy",        f"{best['e']} kWh")
        with c4: st.metric("📊 Efficiency",    f"{best['eff']}/100")
        with c5: st.metric("⚙️ Changeovers",   "1" if best["changeover"] else "0")

        st.markdown(f'<div class="route-box">{make_route_flow(best["steps"])}</div>', unsafe_allow_html=True)
        st.caption("🟡 Yellow=Lathe | 🔵 Blue=Milling | 🟣 Purple=Shared | ⚫ Dark=Universal | ⚙=Changeover")

        clean_steps = [s for s in best["steps"] if s != "--- Machine Changeover ---"]
        if is_complete(clean_steps, features):
            st.success(f"✅ Completeness Verified — All {len(features)} feature(s) covered")
        else:
            st.error("❌ Completeness check failed")

        if calc_mode:
            show_calculator_breakdown(best["calc"], calc_settings, "fa_calc")
            _added, _removed = CALC.job_changes(best["steps"], [r["op"] for r in best["calc"]["rows"]])
            if _added or _removed:
                st.caption("\u270F\uFE0F You changed the job in the operations list: "
                           + (f"added **{', '.join(_added)}** (in the time and cost above)" if _added else "")
                           + ("; " if _added and _removed else "")
                           + (f"taken out **{', '.join(_removed)}** (not in the time and cost above)" if _removed else "")
                           + ". The route shown above is still the planner's.")
            show_v1_split(best["steps"], material, dims, "fa_split_bar")                    # Step 56: what Route Only shows too
        else:
            st.markdown("---")
            st.markdown("#### ⏱️ Time Breakdown — How Total Time is Calculated")
            bd = time_breakdown(best["steps"], material, dims)

            b1,b2,b3,b4,b5 = st.columns(5)
            with b1: st.metric("🔧 Machining Time",      f"{bd['machining_time']} min",
                        help="Pure cutting/forming time per operation")
            with b2: st.metric("🔩 Tool Change Time",    f"{bd['tool_change_time']} min",
                        delta=f"{bd['tool_changes']} × {TOOL_CHANGE_TIME_MIN}min each", delta_color="off")
            with b3: st.metric("📐 Position Change Time",f"{bd['position_change_time']} min",
                        delta=f"{bd['position_changes']} × {POSITION_CHANGE_TIME_MIN}min each", delta_color="off")
            with b4: st.metric("⚙️ Changeover Time",     f"{bd['changeover_time']} min",
                        delta=f"{bd['changeovers']} × {CHANGEOVER_TIME_MIN}min each", delta_color="off")
            with b5: st.metric("📊 TOTAL",               f"{best['t']} min",
                        delta=f"Material factor: {bd['material_factor']}×", delta_color="off")

            st.caption(f"ℹ️ Material factor for **{material}**: {bd['material_factor']}×. "
                       f"Same-tool-group ops do **not** incur tool change time.")
            _own = [(op, v) for op, v in (dims.get("per_op") or {}).items() if v and op in best["steps"]]
            if _own:
                st.caption("📐 Dimensions of their own were used for: " + "; ".join(
                    f"{op} ({', '.join(f'{OPD.LABEL[k].lower()} {x:g}' for k, x in v.items())})" for op, v in _own)
                    + ". Every other operation of this route uses the Part Dimensions.")

            if bd['tool_changes'] > 0 or bd['position_changes'] > 0:
                lines = []
                prev = None
                tc_n = pc_n = 0
                for i, step in enumerate(best["steps"]):
                    if i == 0: prev = step; continue
                    if step in ("Inspection","--- Machine Changeover ---"): prev = step; continue
                    if prev == "--- Machine Changeover ---": prev = step; continue
                    same = any(prev in g and step in g for g in SAME_TOOL_GROUPS)
                    if not same:
                        tc_n += 1
                        lines.append(f"🔩 **Tool Change #{tc_n}:** `{prev}` → `{step}` (+{TOOL_CHANGE_TIME_MIN} min)")
                    if prev not in ("Facing",):
                        pc_n += 1
                        lines.append(f"📐 **Position Change #{pc_n}:** After `{prev}`, re-clamped before `{step}` (+{POSITION_CHANGE_TIME_MIN} min)")
                    prev = step
                if lines:
                    with st.expander(f"ℹ️ Where changes occurred ({bd['tool_changes']} tool + {bd['position_changes']} position)", expanded=True):
                        for line in lines:
                            st.markdown(f"- {line}")

            st.plotly_chart(make_time_bar(bd), use_container_width=True, key="fa_time_bar")

            with st.expander("📋 Per-Operation Time Detail"):
                op_rows = []
                prev = None
                for step in best["steps"]:
                    if step == "--- Machine Changeover ---":
                        op_rows.append({"Operation": "⚙️ Machine Changeover", "Machine": "Setup",
                                        "Base Time": "—", "After Factor": "—",
                                        "Tool Change": "—", "Position Change": "—",
                                        "Note": f"+{CHANGEOVER_TIME_MIN} min penalty"})
                        prev = step; continue
                    mtype = ("Lathe" if step in LATHE_ONLY_OPS
                             else "Milling" if step in MILLING_ONLY_OPS
                             else "Shared" if step not in ("Facing","Inspection") else "Universal")
                    base   = BASE_TIMES.get(step, 5)
                    factor = bd['material_factor']
                    tc_note = "—"
                    if prev and prev != "--- Machine Changeover ---" and step != "Inspection" and prev != "Facing":
                        same = any(prev in g and step in g for g in SAME_TOOL_GROUPS)
                        tc_note = "0 (same tool group)" if same else f"+{TOOL_CHANGE_TIME_MIN} min"
                    pc_note = "—"
                    if prev and prev not in ("Facing","--- Machine Changeover ---") and step != "Inspection":
                        pc_note = f"+{POSITION_CHANGE_TIME_MIN} min"
                    op_rows.append({"Operation": step, "Machine": mtype,
                                    "Base Time": f"{base} min",
                                    "After Factor": f"{round(base*factor,2)} min",
                                    "Tool Change": tc_note, "Position Change": pc_note, "Note": ""})
                    prev = step
                st.dataframe(op_rows, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 💰 Cost Breakdown (₹ INR)")
            from agents import (_COST_PER_MIN, _MATERIAL_COST_FACTOR,
                                _count_setup_events, _op_time,
                                CHANGEOVER_COST_INR, TOOL_CHANGE_COST_INR,
                                POSITION_CHANGE_COST_INR)

            mat_factor = _MATERIAL_COST_FACTOR.get(material, 1.0)
            machining_cost = round(sum(
                _op_time(s, material, dims) * _COST_PER_MIN.get(s, 180)
                for s in best["steps"] if s != "--- Machine Changeover ---"
            ) * mat_factor, 0)
            co_count      = best["steps"].count("--- Machine Changeover ---")
            tc_c, pc_c    = _count_setup_events(best["steps"])
            tool_cost     = round(tc_c * TOOL_CHANGE_COST_INR, 0)
            pos_cost      = round(pc_c * POSITION_CHANGE_COST_INR, 0)
            co_cost       = round(co_count * CHANGEOVER_COST_INR, 0)
            subtotal      = machining_cost + tool_cost + pos_cost + co_cost
            disc_pct      = 15 if batch_size > 500 else 8 if batch_size > 100 else 0
            disc_amt      = round(subtotal * disc_pct / 100, 0)

            cc1,cc2,cc3,cc4 = st.columns(4)
            with cc1: st.metric("🔧 Machining Cost",  f"₹{machining_cost:,.0f}",
                        help=f"Time × rate/min × {mat_factor}× material factor")
            with cc2: st.metric("🔩 Setup Cost",       f"₹{tool_cost+pos_cost:,.0f}",
                        delta=f"Tool:₹{tool_cost:,.0f} + Pos:₹{pos_cost:,.0f}", delta_color="off")
            with cc3: st.metric("⚙️ Changeover Cost",  f"₹{co_cost:,.0f}",
                        delta=f"{co_count} × ₹{CHANGEOVER_COST_INR}", delta_color="off")
            with cc4: st.metric("🏷️ Batch Discount",   f"-₹{disc_amt:,.0f}",
                        delta=f"{disc_pct}% for batch={batch_size}",
                        delta_color="inverse" if disc_pct>0 else "off")

            fig_cost = go.Figure(go.Bar(
                x=["Machining", "Tool & Position Setup", "Changeover", "Discount (-)"],
                y=[machining_cost, tool_cost+pos_cost, co_cost, -disc_amt],
                marker_color=["#3b82f6","#f59e0b","#ef4444","#10b981"],
                text=[f"₹{machining_cost:,.0f}", f"₹{tool_cost+pos_cost:,.0f}",
                      f"₹{co_cost:,.0f}", f"-₹{disc_amt:,.0f}"],
                textposition="outside"
            ))
            fig_cost.add_hline(y=0, line_color="gray", line_width=0.5)
            fig_cost.update_layout(
                title=f"Cost Distribution — Final: ₹{best['c']:,.2f} per unit",
                yaxis_title="Cost (₹ INR)",
                margin=dict(l=40,r=20,t=40,b=30), height=300,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_cost, use_container_width=True, key="fa_cost_bar")
            st.caption(f"Material: **{material}** ({mat_factor}×) | 1 USD = ₹{USD_TO_INR:g} | Batch: {batch_size} units")

        st.markdown("---")
        cl, cr = st.columns([1,1])
        with cl:
            st.markdown("#### Agent Score Breakdown")
            st.plotly_chart(make_radar(best["t"], best["c"], best["e"], best["eff"], scores=best["scores"]),
                            use_container_width=True, key="fa_radar")
            if calc_mode:
                st.caption("Calculator mode: time, cost and energy are scored against the best of the routes above (100 = the best of them).")
        with cr:
            st.markdown("#### Feature → Operation Mapping")
            for feat in features:
                ops  = FEATURE_TO_OPERATIONS[feat]["alternatives"][0]
                m    = get_machine_type(feat)
                icon = "🟡" if m=="Lathe" else "🔵" if m=="Milling" else "🟣"
                st.markdown(f"{icon} **{feat}** ({m}) → {', '.join(ops)}")
            st.markdown("---")
            if calc_mode:
                st.markdown(f"**Labour:** ₹{labor_per_hour:g}/h &nbsp;|&nbsp; **Material cost:** ₹{material_cost:g}")
                st.markdown("**Batch discount:** none (the calculator prices one job)")
            else:
                st.markdown(f"**Batch Discount:** {disc_pct}% (-₹{disc_amt:,.0f})")
                st.markdown(f"**Material Factor:** {mat_factor}×")

        # ── STEP 5: physics + feasibility of the selected route (analysis only) ──
        st.markdown('<div class="step-hdr">STEP 5 — Physics & Feasibility Check of the Selected Route</div>', unsafe_allow_html=True)
        st.caption("Analysis only: the route above was selected by the v1 efficiency score. These numbers describe it and flag problems; "
                   "they do not change which route is chosen.")
        full_view = AV.analyse_route_view(best["steps"], part_av)
        for err in full_view["errors"]:
            st.warning("⚠️ " + err)
        st.markdown("#### 🛡️ Feasibility")
        show_feasibility(full_view["feasibility"], "fa_feas")
        st.markdown("#### 🔬 Cutting Physics")
        show_physics(full_view["physics"], batch_size, "fa_phys", covered_by_rules=full_view["feasibility"] is not None)
        show_assumptions_and_sources(full_view)

        fa_source = "AI-suggested route" + (" (self-corrected)" if "corrected" in best["route"] else "") if best["route"].startswith("Route_LLM") \
            else "Route Builder / NSGA-II"
        if calc_mode:
            st.caption("\U0001F4C4 The route sheet below is the physics model's plan (cutting speed, rpm, feed, times with tool and position changes, the v1 "
                       "model's numbers): its times are not the calculator times above.")
        show_route_sheet(best["steps"], part_av, "fa_sheet", {"source": fa_source})

        st.balloons()

    else:
        st.error("❌ No valid route found. Try different features or strategy.")