import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from utils.apriori_recommender import AprioriRecommender

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="TasteMatch | Food Recommendation System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #fafafa; }
    .food-card {
        background: white;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 18px;
        transition: transform 0.15s ease-in-out;
    }
    .food-card:hover { transform: translateY(-4px); }
    .food-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 2px; }
    .food-meta { color: #777; font-size: 0.85rem; margin-bottom: 6px; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-veg { background: #e3f6e3; color: #1e7d1e; }
    .badge-nonveg { background: #fde3e3; color: #b52121; }
    .badge-vegan { background: #e3edf6; color: #1e5f9e; }
    .rating { color: #f5a623; font-weight: 700; }

    .hero {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff9a5a 100%);
        border-radius: 20px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 { margin: 0 0 4px 0; font-size: 2rem; }
    .hero p { margin: 0; opacity: 0.92; }

    .stat-pill {
        background: white;
        border-radius: 14px;
        padding: 16px 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-top: 4px solid #ff6b6b;
    }
    .stat-pill h2 { margin: 0; font-size: 1.7rem; color: #262626; }
    .stat-pill span { color: #888; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }

    .rec-card {
        background: white;
        border-radius: 18px;
        padding: 0 0 16px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
        margin-bottom: 20px;
        overflow: hidden;
        border: 1px solid #f0f0f0;
    }
    .rec-card img { border-radius: 0; }
    .rec-card-body { padding: 14px 16px 4px 16px; }
    .rec-card-title { font-size: 1.15rem; font-weight: 800; margin-bottom: 10px; }

    .metric-row { display: flex; gap: 8px; padding: 0 16px 4px 16px; }
    .metric-box {
        flex: 1;
        border-radius: 12px;
        padding: 8px 6px;
        text-align: center;
    }
    .metric-support { background: #eaf2ff; }
    .metric-confidence { background: #e6f9ee; }
    .metric-lift { background: #fff2e0; }
    .metric-value { font-size: 1.15rem; font-weight: 800; }
    .metric-support .metric-value { color: #2563eb; }
    .metric-confidence .metric-value { color: #16a34a; }
    .metric-lift .metric-value { color: #d97706; }
    .metric-label { font-size: 0.68rem; color: #888; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }

    .explain-box {
        margin: 10px 16px 0 16px;
        background: #fafafa;
        border-left: 3px solid #ff6b6b;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.85rem;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data
def load_transactions():
    path = Path(__file__).parent / "data" / "transactions.csv"
    tx_df = pd.read_csv(path)
    return [row.split(",") for row in tx_df["items"]]

@st.cache_data
def load_item_images():
    path = Path(__file__).parent / "data" / "item_images.csv"
    return pd.read_csv(path).set_index("item")

@st.cache_resource
def load_apriori_engine(transactions, min_support, min_confidence):
    return AprioriRecommender(transactions, min_support=min_support, min_confidence=min_confidence)

raw_transactions = load_transactions()
item_images = load_item_images()

FALLBACK_IMG = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=500"

def get_item_image(item_name):
    if item_name in item_images.index:
        return item_images.loc[item_name, "image_url"]
    return FALLBACK_IMG

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("🍽️ TasteMatch")
st.sidebar.caption("Apriori-powered food recommendation engine")

page = st.sidebar.radio(
    "Navigate",
    ["🎯 Recommendation System", "📈 Rule Explorer"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Apriori Settings")
min_support = st.sidebar.slider("Min support", 0.05, 0.6, 0.20, 0.05)
min_confidence = st.sidebar.slider("Min confidence", 0.1, 1.0, 0.5, 0.05)
st.sidebar.caption(
    "**Support** = how often the itemset appears.\n\n"
    "**Confidence** = P(consequent | antecedent).\n\n"
    "**Lift** = how much more likely than random chance."
)

apriori_engine = load_apriori_engine(raw_transactions, min_support, min_confidence)


# ----------------------------------------------------------------------
# Helper: render a recommendation card with support/confidence/lift
# ----------------------------------------------------------------------
def render_recommendation_card(item_name, support, confidence, lift, base_item):
    img = get_item_image(item_name)
    st.markdown('<div class="rec-card">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown(
        f"""
        <div class="rec-card-body">
            <div class="rec-card-title">🍴 {item_name}</div>
        </div>
        <div class="metric-row">
            <div class="metric-box metric-support">
                <div class="metric-value">{support*100:.0f}%</div>
                <div class="metric-label">Support</div>
            </div>
            <div class="metric-box metric-confidence">
                <div class="metric-value">{confidence*100:.0f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric-box metric-lift">
                <div class="metric-value">{lift:.2f}x</div>
                <div class="metric-label">Lift</div>
            </div>
        </div>
        <div class="explain-box">
            {confidence*100:.0f}% of customers who ordered <b>{base_item}</b> also ordered <b>{item_name}</b>
            — {lift:.2f}× more likely than by chance.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ========================================================================
# PAGE 1: RECOMMENDATION SYSTEM (main / hero page)
# ========================================================================
if page == "🎯 Recommendation System":
    st.markdown(
        """
        <div class="hero">
            <h1>🍽️ Smart Food Recommendation System</h1>
            <p>Powered by the Apriori market-basket algorithm — trained on real order transactions
            to answer "customers who ordered this also ordered..."</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_orders = len(raw_transactions)
    n_items = len(apriori_engine.all_items)
    n_itemsets = len(apriori_engine.frequent_itemsets)
    n_rules = len(apriori_engine.rules)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [n_orders, n_items, n_itemsets, n_rules],
        ["Orders Analyzed", "Unique Items", "Frequent Itemsets", "Association Rules"],
    ):
        with col:
            st.markdown(
                f'<div class="stat-pill"><h2>{val}</h2><span>{label}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### ")
    st.markdown("### 🛒 What did the customer order?")
    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        item_choice = st.selectbox(
            "Select an item to get recommendations for:",
            apriori_engine.all_items,
            label_visibility="collapsed",
        )
    with sel_col2:
        st.image(get_item_image(item_choice), width=140)

    st.caption(f"Selected: **{item_choice}**")

    recs = apriori_engine.recommend(item_choice)

    st.markdown(f"## 🎯 Recommended with **{item_choice}**")

    if recs.empty:
        st.warning(
            f"No strong association rules found for **{item_choice}** at "
            f"support ≥ {min_support:.2f} / confidence ≥ {min_confidence:.2f}. "
            "Try lowering the thresholds in the sidebar."
        )
        st.info("Here are the most popular items overall instead:")
        popularity = apriori_engine.encoded_df.mean().sort_values(ascending=False).head(4)
        cols = st.columns(len(popularity))
        for col, (item, freq) in zip(cols, popularity.items()):
            with col:
                st.image(get_item_image(item), use_container_width=True)
                st.markdown(f"**{item}**")
                st.caption(f"Ordered in {freq*100:.0f}% of transactions")
    else:
        cols = st.columns(min(len(recs), 3) or 1)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % len(cols)]:
                render_recommendation_card(
                    row["recommended_items"], row["support"], row["confidence"], row["lift"], item_choice
                )

        st.markdown("### 📊 Confidence & Lift Comparison")
        chart_df = recs.copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_df["recommended_items"], y=chart_df["confidence"] * 100,
            name="Confidence (%)", marker_color="#16a34a", yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=chart_df["recommended_items"], y=chart_df["lift"],
            name="Lift (x)", marker_color="#d97706", yaxis="y2", mode="lines+markers",
            line=dict(width=3),
        ))
        fig.update_layout(
            yaxis=dict(title="Confidence (%)", range=[0, 100]),
            yaxis2=dict(title="Lift", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.15),
            margin=dict(t=30),
        )
        st.plotly_chart(fig, use_container_width=True)


# ========================================================================
# PAGE 2: RULE EXPLORER
# ========================================================================
elif page == "📈 Rule Explorer":
    st.title("📈 Association Rule Explorer")
    st.caption("Full transparency into how the Apriori engine mines recommendations")

    with st.expander("📦 Raw order transactions used for training", expanded=False):
        st.dataframe(
            pd.DataFrame({
                "order_id": range(1, len(raw_transactions) + 1),
                "items": [", ".join(t) for t in raw_transactions],
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("### 🔢 Frequent Itemsets")
    fi = apriori_engine.frequent_itemsets.copy()
    fi["itemsets"] = fi["itemsets"].apply(lambda x: ", ".join(sorted(x)))
    st.dataframe(
        fi.rename(columns={"itemsets": "Itemset", "support": "Support"})
        .style.format({"Support": "{:.2f}"})
        .background_gradient(cmap="Oranges", subset=["Support"]),
        use_container_width=True, hide_index=True,
    )

    fig_fi = px.bar(
        fi.head(12), x="itemsets" if "itemsets" in fi.columns else "Itemset", y="support",
        title="Top Frequent Itemsets by Support", color="support", color_continuous_scale="Oranges",
    )
    fig_fi.update_layout(xaxis_tickangle=-40, showlegend=False, xaxis_title="Itemset", yaxis_title="Support")
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("### 📏 Association Rules (Support / Confidence / Lift)")
    if apriori_engine.rules.empty:
        st.warning("No rules at the current thresholds — lower min support / confidence in the sidebar.")
    else:
        rules_display = apriori_engine.rules.copy()
        rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ", ".join(sorted(x)))
        rules_display = rules_display[["antecedents", "consequents", "support", "confidence", "lift"]]
        rules_display.columns = ["If customer orders", "Then also recommend", "Support", "Confidence", "Lift"]

        st.dataframe(
            rules_display.style.format({"Support": "{:.2f}", "Confidence": "{:.2f}", "Lift": "{:.2f}"})
            .background_gradient(cmap="Greens", subset=["Confidence"])
            .background_gradient(cmap="Oranges", subset=["Lift"]),
            use_container_width=True, hide_index=True,
        )

        st.markdown("### 🕸️ Item Association Network")
        st.caption("Line thickness = lift · Line color intensity = confidence")

        items_in_rules = sorted(set(rules_display["If customer orders"]) | set(rules_display["Then also recommend"]))
        n = len(items_in_rules)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {item: (np.cos(a), np.sin(a)) for item, a in zip(items_in_rules, angles)}

        edge_traces = []
        for _, r in rules_display.iterrows():
            x0, y0 = pos[r["If customer orders"]]
            x1, y1 = pos[r["Then also recommend"]]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines",
                    line=dict(width=max(r["Lift"], 1), color=f"rgba(255,107,107,{min(r['Confidence']+0.2,1)})"),
                    hoverinfo="text",
                    text=f"{r['If customer orders']} → {r['Then also recommend']}<br>Conf: {r['Confidence']:.2f}, Lift: {r['Lift']:.2f}",
                    showlegend=False,
                )
            )

        node_x = [pos[i][0] for i in items_in_rules]
        node_y = [pos[i][1] for i in items_in_rules]
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=items_in_rules, textposition="top center",
            marker=dict(size=22, color="#ff9a5a", line=dict(width=2, color="white")),
            showlegend=False,
        )

        fig_net = go.Figure(data=edge_traces + [node_trace])
        fig_net.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(t=10, b=10, l=10, r=10), height=500,
        )
        st.plotly_chart(fig_net, use_container_width=True)



st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • TasteMatch Recommendation Engine")
