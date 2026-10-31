import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.scenarios import SCENARIOS
from modules.solve_2x2 import graphical_data
from modules.solver import solve_game
from utils.data_validator import validate_matrix

# ---------------------------------------------------------
# 1. Page configuration & styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="EquiMatrix - Game Theory Solver",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    html, body, [class*="css"], .stMarkdown, p, div, h1, h2, h3, button, input {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .main-title { font-size: 2.6rem; font-weight: 800; color: #1E3A8A; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #4B5563; text-align: center; font-style: italic; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚖️ EQUIMATRIX OPTIMIZATION SOLVER</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Applications of Matrices in Two-Player Zero-Sum Game Theory • Linear Algebra Project</div>',
    unsafe_allow_html=True,
)

def short(names):
    """Short axis labels: the part before the colon (e.g. "X1: Patrol" -> "X1")."""
    return [n.split(":")[0] for n in names]


# ---------------------------------------------------------
# 2. Sidebar: choose the game
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    mode = st.radio("Input mode", ["Real-world scenario library", "Custom m × n matrix"])

    scenario = None
    if mode.startswith("Real-world"):
        titles = [s.title for s in SCENARIOS]
        chosen = st.selectbox("Choose scenario:", titles)
        scenario = SCENARIOS[titles.index(chosen)]
        A0 = scenario.A.copy()
        row_names, col_names = list(scenario.row_names), list(scenario.col_names)
    else:
        m_in = st.number_input("Rows m (Player 1 strategies):", min_value=2, max_value=8, value=3, step=1)
        n_in = st.number_input("Columns n (Player 2 strategies):", min_value=2, max_value=8, value=3, step=1)
        m_in, n_in = int(m_in), int(n_in)
        A0 = np.zeros((m_in, n_in), dtype=float)
        row_names = [f"X{i + 1}" for i in range(m_in)]
        col_names = [f"Y{j + 1}" for j in range(n_in)]
        st.caption("Player 1 (rows) maximises the payoff a_ij, Player 2 (columns) minimises it.")

m_rows, n_cols = A0.shape

# ---------------------------------------------------------
# Helper plots
# ---------------------------------------------------------
def plot_strategy_distribution(probs, player_name, color_scale, labels):
    probs = np.asarray(probs, dtype=float)
    labels = list(labels)
    if len(probs) != len(labels):
        raise ValueError(f"{player_name}: probability count mismatch.")

    df = pd.DataFrame({
        "Strategy": labels,
        "Probability": probs,
        "Label": [f"{value:.4f}" for value in probs],
    })
    fig = px.bar(
        df, x="Strategy", y="Probability", text="Label",
        title=f"{player_name} Optimal Strategy Distribution",
        color="Probability", color_continuous_scale=color_scale,
        category_orders={"Strategy": labels},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        font_family="Times New Roman", title_font_size=16, showlegend=False,
        yaxis=dict(range=[0, 1.25], title="Probability", tickformat=".2f"),
        xaxis=dict(type="category", categoryorder="array", categoryarray=labels),
        coloraxis_showscale=False, margin=dict(t=65, b=50),
    )
    return fig


def plot_envelope(g, player, labels_other):
    """Graphical method: lines for each opponent pure strategy and the lower / upper envelope."""
    fig = go.Figure()
    for k, line in enumerate(g["lines"]):
        fig.add_trace(go.Scatter(x=g["t"], y=line, mode="lines", name=labels_other[k], line=dict(width=1.5)))
    env_name = "Lower envelope (Player 1 maximises)" if player == "row" else "Upper envelope (Player 2 minimises)"
    fig.add_trace(go.Scatter(x=g["t"], y=g["envelope"], mode="lines", name=env_name,
                             line=dict(width=4, dash="dash", color="crimson")))
    fig.add_trace(go.Scatter(x=[g["t_opt"]], y=[g["v_opt"]], mode="markers+text",
                             text=[f"t*={g['t_opt']:.3f}, v={g['v_opt']:.3f}"], textposition="top center",
                             marker=dict(size=12, color="crimson"), name="Optimum"))
    xlabel = "Probability of Player 1's first strategy" if player == "row" else "Probability of Player 2's first strategy"
    fig.update_layout(font_family="Times New Roman", xaxis_title=xlabel, yaxis_title="Expected payoff",
                      title="Graphical method (envelope of expected payoffs)", margin=dict(t=60))
    return fig


# ---------------------------------------------------------
# 3. Matrix input and computation
# ---------------------------------------------------------
st.subheader("📝 Payoff Matrix A = (aᵢⱼ)")
if scenario is not None:
    st.caption(f"Payoff unit: {scenario.unit}. Rows = {scenario.row_player}, columns = {scenario.col_player}. "
               "You may edit the numbers.")

default_df = pd.DataFrame(A0, columns=col_names, index=row_names)
edited_df = st.data_editor(
    default_df,
    key=f"editor_{mode}_{scenario.key if scenario else 'custom'}_{m_rows}x{n_cols}",
    num_rows="fixed",
)
solve_btn = st.button("🚀 Calculate Equilibrium Solution", type="primary")

ok, msg, A = validate_matrix(edited_df.to_numpy())
if not ok:
    st.error(msg)
else:
    signature = (A.shape, tuple(A.flatten().tolist()))
    if solve_btn:
        try:
            st.session_state["report"] = (signature, solve_game(A))
            st.balloons()
        except Exception as exc:  # show the real reason instead of failing silently
            st.session_state.pop("report", None)
            st.error(f"The solver failed: {exc}")

    stored = st.session_state.get("report")
    if stored is None or stored[0] != signature:
        st.info("Press **Calculate Equilibrium Solution** to solve the current matrix.")
    else:
        rep = stored[1]
        sp = rep.saddle
        st.markdown("---")
        st.header("🎯 Equilibrium Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Value of the game v", f"{rep.value:.4f}")
        c2.metric("Maximin (lower value)", f"{sp['maximin']:.4f}")
        c3.metric("Minimax (upper value)", f"{sp['minimax']:.4f}")
        c4.metric("x* ᵀ A y*", f"{rep.verification['xAy']:.4f}")
        st.caption(f"Method used: **{rep.method}**")

        # --- Step 1: maximin / minimax
        st.markdown("#### Step 1 - Maximin and Minimax principles")
        table = pd.DataFrame(A, index=row_names, columns=col_names)
        table["Row min"] = A.min(axis=1)
        table.loc["Column max"] = list(A.max(axis=0)) + [np.nan]
        st.dataframe(table)
        if sp["has_saddle_point"]:
            pts = ", ".join(f"({row_names[i].split(':')[0]}, {col_names[j].split(':')[0]})" for i, j in sp["saddle_points"])
            st.success(f"✅ Maximin = Minimax = {sp['value']:.4f}. Pure saddle point(s): {pts}.")
        else:
            st.warning(f"⚠️ Maximin ({sp['maximin']:.4f}) < Minimax ({sp['minimax']:.4f}): no pure saddle point, "
                       "so optimal play requires mixed strategies (Minimax Theorem).")

        # --- Step 2: dominance
        with st.expander("Step 2 - Elimination of strictly dominated strategies"):
            for line in rep.dominance["steps"]:
                st.write("• " + line)
            red = rep.dominance["reduced"]
            st.write(f"Reduced matrix ({red.shape[0]} × {red.shape[1]}):")
            st.dataframe(pd.DataFrame(
                red,
                index=[row_names[i] for i in rep.dominance["row_idx"]],
                columns=[col_names[j] for j in rep.dominance["col_idx"]],
            ))

        # --- Step 3: solution
        st.markdown("#### Step 3 - Optimal strategies")
        if rep.two_by_two is not None:
            for line in rep.two_by_two.steps:
                st.write("• " + line)

        col1, col2 = st.columns(2)
        col1.plotly_chart(plot_strategy_distribution(rep.x, "Player 1", "Blues", short(row_names)))
        col2.plotly_chart(plot_strategy_distribution(rep.y, "Player 2", "Tealgrn", short(col_names)))

        t1, t2 = st.columns(2)
        t1.dataframe(pd.DataFrame({"Strategy": row_names, "x*": rep.x.round(6)}))
        t2.dataframe(pd.DataFrame({"Strategy": col_names, "y*": rep.y.round(6)}))

        # --- Step 4: verification
        st.markdown("#### Step 4 - Verification of optimality")
        ver = rep.verification
        v1, v2 = st.columns(2)
        v1.write("Player 1's expected payoff against each pure column, (xᵀA)ⱼ — must all be ≥ v:")
        v1.dataframe(pd.DataFrame({"Column": col_names, "(xᵀA)ⱼ": ver["col_payoffs"].round(6)}))
        v2.write("Player 1's expected payoff of each pure row against y*, (Ay)ᵢ — must all be ≤ v:")
        v2.dataframe(pd.DataFrame({"Row": row_names, "(Ay)ᵢ": ver["row_payoffs"].round(6)}))
        if ver["is_optimal"]:
            st.success(f"✅ Verified: min (xᵀA)ⱼ = {ver['p1_guarantee']:.4f} ≥ v = {rep.value:.4f} ≥ "
                       f"max (Ay)ᵢ = {ver['p2_guarantee']:.4f}, and xᵀAy = {ver['xAy']:.4f}.")
        else:
            st.error("❌ The verification failed: (x*, y*) is not an optimal pair.")

        # --- Graphical method (2 x n or m x 2)
        gdata = graphical_data(A)
        if gdata:
            with st.expander("Graphical method (2 × n or m × 2 games)"):
                if "row" in gdata:
                    st.plotly_chart(plot_envelope(gdata["row"], "row", col_names))
                if "col" in gdata:
                    st.plotly_chart(plot_envelope(gdata["col"], "col", row_names))

        # --- Cross-check
        st.markdown("#### Step 5 - Cross-check of the value of the game")
        cross = pd.DataFrame({
            "Method": ["Main solution", "LP (SciPy HiGHS)", "Own simplex implementation"],
            "Value v": [rep.value, rep.lp.value, rep.simplex.value],
            "Optimal pair verified": [ver["is_optimal"], rep.lp_verification["is_optimal"],
                                      rep.simplex_verification["is_optimal"]],
        })
        st.dataframe(cross)
        if rep.values_agree:
            st.success("✅ All methods give the same value of the game (Minimax Theorem).")
        else:
            st.error("❌ The methods disagree on the value; check the input matrix.")
