import streamlit as st
from optimizer import run_optimization_tool, run_route_optimizer_ui



# ✅ This must be first Streamlit command, inside main guard
def main():
    st.set_page_config(page_title="Freight Pricing & Optimization Tool", layout="wide")

    # ⬅️ Import all optimizers and pricing tools
    from pricing_engine import run_pricing_simulator
    from optimizer import run_optimization_tool, run_route_optimizer_ui

    # --- Custom CSS Styling ---
    st.markdown(
        """
        <style>
        .title-container {
            display: flex;
            align-items: center;
        }
        .title-container img {
            margin-right: 10px;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.9em;
            color: #999;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- App Title ---
    st.markdown(
        '<div class="title-container">'
        '<img src="https://img.icons8.com/emoji/48/delivery-truck.png"/>'
        '<h1> Freight Pricing & Optimization Tool</h1>'
        '</div>',
        unsafe_allow_html=True
    )

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "🚛 Dynamic Pricing Simulator",
        "📊 Shipment Optimizer",
        "🗺️ Route Optimizer (with Fuel Stops)"
    ])

    with tab1:
        run_pricing_simulator()

    with tab2:
        run_optimization_tool()

    with tab3:
        run_route_optimizer_ui()

    # --- Footer ---
    st.markdown("""
    <div class="footer">
        🚀 Built by <a href="https://www.linkedin.com/in/cheranratnam/" target="_blank">Cheran Ratnam</a> |
        📊 Data Source: <a href="https://faf.ornl.gov/faf5" target="_blank">FAF5 Freight Dataset</a> |
        🌐 <a href="https://www.linkedin.com/in/cheranratnam/" target="_blank">LinkedIn</a> |
        💼 <a href="https://github.com/cheranratnam87/logistics" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ✅ Run only when executed as main
if __name__ == "__main__":
    main() 
