# ✅ MUST be the first Streamlit command
import streamlit as st
st.set_page_config(page_title="Freight Pricing & Optimization Tool", layout="wide")

from optimizer import run_optimization_tool, run_multi_destination_route_optimizer, load_data, load_cng_stations
from pricing_engine import run_pricing_simulator

# Preload data to reduce lag when switching tabs
if "preloaded_data" not in st.session_state:
    st.session_state.preloaded_data = {
        "freight_data": load_data(),
        "cng_stations": load_cng_stations()
    }

def main():
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

    # Use session state to remember the active tab
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Dynamic Pricing Simulator"

    tabs = {
        "Dynamic Pricing Simulator": run_pricing_simulator,
        "Shipment Optimizer": lambda: run_optimization_tool(st.session_state.preloaded_data["freight_data"]),
        "Route Optimizer (with Fuel Stops)": lambda: run_multi_destination_route_optimizer(
            st.session_state.preloaded_data["freight_data"],
            st.session_state.preloaded_data["cng_stations"]
        ),
    }

    selected_tab = st.sidebar.radio("Select a Tool", list(tabs.keys()), index=list(tabs.keys()).index(st.session_state.active_tab))
    st.session_state.active_tab = selected_tab

    # Run the selected tab's function
    tabs[selected_tab]()

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