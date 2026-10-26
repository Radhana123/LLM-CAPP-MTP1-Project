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
#   - Pareto front 3D + 2D charts
#   - Per-operation detail table

import streamlit as st
import sys, os, time
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load .env from week6 folder
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week1")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week2")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week3")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week4")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week5")))

from tokenizer import tokenize
from feature_vocab import GEOMETRY_FEATURES, FEATURE_TO_OPERATIONS, get_machine_type
from agents import (time_agent, cost_agent, energy_agent, efficiency_agent,
                    time_breakdown, TOOL_CHANGE_TIME_MIN,
                    POSITION_CHANGE_TIME_MIN, CHANGEOVER_TIME_MIN,
                    SAME_TOOL_GROUPS)
from nsga2 import run_nsga2
from fsm_validator import validate_sequence
from self_corrector import self_correct
from precedence_graph import PRECEDENCE_EDGES
from route_builder import (generate_valid_routes, is_complete,
                           LATHE_ONLY_OPS, MILLING_ONLY_OPS)

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


def make_radar(t, c, e, eff):
    cats = ['Time', 'Cost', 'Energy', 'Efficiency', 'Time']
    vals = [max(0,100-t*1.2), max(0,100-c*0.005), max(0,100-e*15), eff,
            max(0,100-t*1.2)]
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


def make_pareto_charts(data_list, best_route_name, key_prefix):
    """Draw 3D + 2D Pareto front charts. Returns True if a chart was drawn,
    False if there weren't enough genuinely distinct points to plot --
    caller should show its own message in that case rather than leaving
    silent empty space."""
    # Route Builder kabhi kabhi same operation-set ke multiple step-order
    # variants deta hai -- unka Time/Cost/Energy bilkul identical hota hai
    # (order se in metrics pe farak nahi padta), isliye wo chart mein
    # bilkul ek hi (x,y,z) point pe overlap ho jaate the aur unke text
    # labels ek dusre ke upar garbled ho jaate the. Ab metrics ke hisaab se
    # dedupe karte hain -- ek hi jagah ka sirf ek representative point
    # dikhega, hover mein baaki equivalent routes ka count mil jaayega.
    seen = {}
    deduped = []
    for r in data_list:
        key = (r["t"], r["c"], r["e"])
        if key not in seen:
            seen[key] = r
            deduped.append(r)
        else:
            seen[key].setdefault("_equivalent_count", 1)
            seen[key]["_equivalent_count"] += 1
            # Agar best route in duplicates mein hai, to representative
            # point ka naam bhi best route ka rakho taaki highlight sahi jagah lage
            if r["route"] == best_route_name:
                seen[key]["route"] = best_route_name
    data_list = deduped

    if len(data_list) < 2:
        return False

    times    = [r["t"] for r in data_list]
    costs    = [r["c"] for r in data_list]
    energies = [r["e"] for r in data_list]
    effs     = [r["eff"] for r in data_list]
    hover    = [
        f"<b>{r['route']}</b>" +
        (f" (+{r['_equivalent_count']-1} equivalent route(s))" if r.get("_equivalent_count", 1) > 1 else "") +
        f"<br>Machine: {r['label']}<br>"
        f"Time: {r['t']} min<br>Cost: ₹{r['c']:,.0f}<br>"
        f"Energy: {r['e']} kWh<br>Efficiency: {r['eff']}/100"
        for r in data_list
    ]

    # 3D — chhote, clean dots; koi per-point text label chart pe nahi
    # (sirf hover pe) taaki bahut saare points hone par bhi clutter na ho
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=times, y=costs, z=energies,
        mode='markers',
        hovertext=hover, hoverinfo='text',
        marker=dict(size=6, color=effs, colorscale='RdYlGn',
                    colorbar=dict(title="Efficiency"),
                    showscale=True, opacity=0.85,
                    line=dict(color='white', width=0.5)),
        name="Routes"
    ))
    best = next((r for r in data_list if r["route"] == best_route_name), None)
    if best:
        fig3d.add_trace(go.Scatter3d(
            x=[best["t"]], y=[best["c"]], z=[best["e"]],
            mode='markers', hovertext=[f"<b>⭐ OPTIMAL — {best['route']}</b><br>Time:{best['t']}min Cost:₹{best['c']:,.0f}"],
            hoverinfo='text',
            marker=dict(size=12, color='gold', symbol='diamond',
                        line=dict(color='orange', width=2)),
            name="Optimal"
        ))
    fig3d.update_layout(
        scene=dict(xaxis_title="Time (min)", yaxis_title="Cost (₹)",
                   zaxis_title="Energy (kWh)", bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0,r=0,t=20,b=0), height=420,
        paper_bgcolor='rgba(0,0,0,0)', showlegend=True
    )
    st.plotly_chart(fig3d, use_container_width=True, key=f"{key_prefix}_3d")

    # 2D — same clean-dot style
    st.markdown("##### 2D View: Time vs Cost  *(bubble size = Energy)*")
    fig2d = go.Figure()
    for i, r in enumerate(data_list):
        is_best = r["route"] == best_route_name
        fig2d.add_trace(go.Scatter(
            x=[r["t"]], y=[r["c"]],
            mode='markers',
            hovertext=[hover[i]], hoverinfo='text',
            marker=dict(size=max(10, r["e"]*10) if is_best else max(8, r["e"]*8),
                        color='gold' if is_best else '#3b82f6',
                        symbol='diamond' if is_best else 'circle',
                        line=dict(color='orange' if is_best else 'white', width=1.5),
                        opacity=0.85),
            showlegend=False
        ))
    fig2d.update_layout(
        xaxis_title="Time (min)  →  Lower is Better",
        yaxis_title="Cost (₹)  →  Lower is Better",
        margin=dict(l=40,r=20,t=10,b=40), height=300,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,252,1)'
    )
    st.plotly_chart(fig2d, use_container_width=True, key=f"{key_prefix}_2d")
    return True


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Part Configuration")
    st.markdown("---")

    material = st.selectbox("🧱 Material",
        ["Aluminum","Steel","Brass","Copper","Titanium","Plastic","Cast Iron"])

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

    if True:
        uploaded_image = st.file_uploader("Upload a 2D engineering drawing / sketch",
            type=["png", "jpg", "jpeg"])

        if uploaded_image is not None:
            if st.button("🔍 Extract Features from Image", use_container_width=True):
                with st.spinner("Analyzing image with AI vision model..."):
                    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week1")))
                    from image_feature_extractor import extract_features_from_image
                    img_bytes = uploaded_image.getvalue()
                    img_format = uploaded_image.type.split("/")[-1]
                    extraction_info = extract_features_from_image(img_bytes, image_format=img_format)
                st.session_state["extraction_result"] = extraction_info
                st.session_state["uploaded_image_bytes"] = img_bytes
                st.session_state["confirmed_features"] = extraction_info["features"]

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
                    st.warning(f"⚠️ Ignored unrecognized terms: {', '.join(extraction_info['rejected_features'])}")
                if not extraction_info["features"]:
                    with st.expander("🔍 Debug: What did the AI actually say? (nothing was detected)"):
                        st.code(extraction_info.get("raw_response", "(empty)"), language="text")

                st.markdown("**✏️ Confirm or edit detected features:**")
                features = st.multiselect("Features (edit if AI missed/misread something)",
                    GEOMETRY_FEATURES, default=extraction_info["features"], key="confirmed_features")
                # session_state se sync karo — rerun pe multiselect reset ho jaata hai
                if not features and "confirmed_features" in st.session_state:
                    features = st.session_state["confirmed_features"]
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
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            part_diameter = st.number_input("Diameter (mm)", min_value=1.0,
                max_value=500.0, value=25.0, step=1.0)
            part_depth = st.number_input("Depth (mm)", min_value=1.0,
                max_value=300.0, value=20.0, step=1.0)
        with col_d2:
            part_length = st.number_input("Length (mm)", min_value=1.0,
                max_value=1000.0, value=50.0, step=1.0)
            part_width = st.number_input("Width (mm)", min_value=1.0,
                max_value=500.0, value=20.0, step=1.0)
        dims = {"diameter": part_diameter, "length": part_length,
                "depth": part_depth, "width": part_width}
        st.markdown("---")
    else:
        dims = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}

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
    run_btn = st.button("🚀 Generate Process Plan", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("LLM-CAPP v2.0 | IIT Kharagpur SRIC")
    st.caption("Prof. Sankha Deb | Summer 2026")


# ── Main Page ─────────────────────────────────────────────────
st.markdown('<p class="hero-title">⚙️ LLM-Based Intelligent Process Planning</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Dynamic Route Builder · Machine-Aware Grouping · NSGA-II · INR Costing · Pareto Front</p>', unsafe_allow_html=True)

if not run_btn:
    cols = st.columns(5)
    for col, (n, l) in zip(cols, [
        (str(len(GEOMETRY_FEATURES)),"Features"), ("41","Operations"),
        ("200","Dataset"), (str(len(PRECEDENCE_EDGES)),"Precedence Rules"), ("₹95.595","Per USD")
    ]):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{n}</div>'
                        f'<div class="stat-lbl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🏭 Machine-Aware Routing")
        st.markdown("Lathe ops 🟡 grouped together, Milling ops 🔵 grouped together. Changeover cost/time penalty applied automatically.")
    with c2:
        st.markdown("#### ⏱️ Realistic Time Model")
        st.markdown(f"4 components: Machining + Tool Change (+{TOOL_CHANGE_TIME_MIN}min) + Position Change (+{POSITION_CHANGE_TIME_MIN}min) + Changeover (+{CHANGEOVER_TIME_MIN}min). Multi-pass depth calculation included.")
    with c3:
        st.markdown("#### 💰 INR Costing")
        st.markdown("All costs in ₹ INR (1 USD = ₹95.595). Cost = machining time × rate/min. Batch discount: >100 → 8% off, >500 → 15% off.")
    st.caption("👈 Configure part in sidebar → click **Generate Process Plan**")

else:
    # Image upload mode mein features session_state se lo
    if not features and "confirmed_features" in st.session_state:
        features = st.session_state["confirmed_features"]

    if not features:
        st.error("❌ Please select at least one feature!")
        st.stop()

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

            st.markdown("---")
            st.markdown("#### 📈 Pareto Front")

            ro_data = []
            for ind in valid_routes:
                rt   = time_agent(ind.steps, material)
                rc   = cost_agent(ind.steps, material, batch_size)
                re_  = energy_agent(ind.steps, material)
                reff = efficiency_agent(rt, rc, re_)
                ro_data.append({"route": ind.route_name, "steps": ind.steps,
                                "t": rt, "c": rc, "e": re_, "eff": reff,
                                "label": get_route_label(ind.steps)})

            # Kuch features (Hole, Thread, Face, Boss, Keyway, Groove,
            # Turning, Contour_3D) ke multiple ALTERNATIVE operation-chains
            # hain (e.g. Hole = Drilling, ya Drilling+Reaming, ya
            # Drilling+Boring -- teeno alag). Route Builder inhe already
            # explore karta hai jab max_routes zyada diya jaaye, aur har
            # alternative ka Time/Cost/Energy genuinely alag hota hai --
            # isliye yahan explicitly zyada candidates maang ke Pareto
            # front mein ye asli diversity bhi shaamil kar lete hain
            # (koi fake/duplicate data nahi, sirf jo already exist karta
            # hai use surface karte hain).
            if len(ro_data) < 5:
                extra_routes = generate_valid_routes(features, max_routes=50, machine_preference=pref_value)
                for r in extra_routes:
                    clean_check = [s for s in r["steps"] if s != "--- Machine Changeover ---"]
                    if not is_complete(clean_check, features):
                        continue
                    if not validate_sequence(clean_check)["valid"]:
                        continue
                    rt   = time_agent(r["steps"], material)
                    rc   = cost_agent(r["steps"], material, batch_size)
                    re_  = energy_agent(r["steps"], material)
                    reff = efficiency_agent(rt, rc, re_)
                    ro_data.append({"route": f"Route_Alt_{len(ro_data)+1}", "steps": r["steps"],
                                    "t": rt, "c": rc, "e": re_, "eff": reff,
                                    "label": get_route_label(clean_check)})

            if len(ro_data) >= 2:
                ro_data.sort(key=lambda x: -x["eff"])
                drew_chart = make_pareto_charts(ro_data, best_route_name, "ro_pareto")
                if not drew_chart:
                    st.info("All candidate routes for this feature combination converge to the same Time/Cost/Energy — nothing distinct to plot on a Pareto chart.")
            elif len(ro_data) == 1:
                st.info("Only one distinct valid route found for this feature combination — not enough points for a Pareto chart.")

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

    for ind in valid_routes:
        t   = time_agent(ind.steps, material, dims)
        c   = cost_agent(ind.steps, material, batch_size, dims)
        e   = energy_agent(ind.steps, material, dims)
        eff = efficiency_agent(t, c, e)
        has_co = "--- Machine Changeover ---" in ind.steps

        agent_data.append({
            "Route": ind.route_name,
            "Machine Type": get_route_label(ind.steps),
            "Changeover": "Yes ⚙️" if has_co else "No",
            "Time (min)": t, "Cost (₹)": c,
            "Energy (kWh)": e, "Efficiency": eff
        })

        if eff > best_eff:
            best_eff = eff
            best = {"route": ind.route_name, "steps": ind.steps,
                    "t": t, "c": c, "e": e, "eff": eff, "changeover": has_co}

    st.dataframe(agent_data, use_container_width=True, hide_index=True)

    if len(agent_data) >= 2:
        st.markdown("#### 📈 Pareto Front — Time vs Cost vs Energy")
        st.caption("Each point = one Pareto-optimal route. ⭐ = selected optimal.")
        fa_data = [{"route": r["Route"], "t": r["Time (min)"], "c": r["Cost (₹)"],
                    "e": r["Energy (kWh)"], "eff": r["Efficiency"],
                    "label": r["Machine Type"]} for r in agent_data]
        drew_chart = make_pareto_charts(fa_data, best["route"] if best else "", "fa_pareto")
        if not drew_chart:
            st.info("All candidate routes converge to the same Time/Cost/Energy — nothing distinct to plot on a Pareto chart.")

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
        st.caption(f"Material: **{material}** ({mat_factor}×) | 1 USD = ₹95.595 | Batch: {batch_size} units")

        st.markdown("---")
        cl, cr = st.columns([1,1])
        with cl:
            st.markdown("#### Agent Score Breakdown")
            st.plotly_chart(make_radar(best["t"], best["c"], best["e"], best["eff"]),
                            use_container_width=True, key="fa_radar")
        with cr:
            st.markdown("#### Feature → Operation Mapping")
            for feat in features:
                ops  = FEATURE_TO_OPERATIONS[feat]["alternatives"][0]
                m    = get_machine_type(feat)
                icon = "🟡" if m=="Lathe" else "🔵" if m=="Milling" else "🟣"
                st.markdown(f"{icon} **{feat}** ({m}) → {', '.join(ops)}")
            st.markdown("---")
            st.markdown(f"**Batch Discount:** {disc_pct}% (-₹{disc_amt:,.0f})")
            st.markdown(f"**Material Factor:** {mat_factor}×")

        st.balloons()

    else:
        st.error("❌ No valid route found. Try different features or strategy.")