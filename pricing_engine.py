import streamlit as st
import pandas as pd
import ast
from geopy.distance import geodesic

from utils import (
    get_commodity_class,
    get_allowed_trucks,
    estimate_price_with_breakdown
)

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_freight_data.csv")
    df["origin_city"] = df["origin_city"].str.strip()
    df["destination_city"] = df["destination_city"].str.strip()
    return df

df = load_data()

# --- Coordinate Parsing ---
def parse_coords_safe(coord_str):
    try:
        return ast.literal_eval(coord_str)
    except:
        return None

# --- Build city → coord map ---
city_coords_map = {}
for _, row in df.iterrows():
    city_coords_map[row["origin_city"]] = parse_coords_safe(row.get("origin_coords"))
    city_coords_map[row["destination_city"]] = parse_coords_safe(row.get("destination_coords"))

all_cities = sorted(set(df["origin_city"]).union(df["destination_city"]))
valid_cities = [c for c in all_cities if city_coords_map.get(c) is not None]

# --- Distance functions ---
@st.cache_data
def get_avg_distance(df, origin, destination):
    match = df[(df["origin_city"] == origin) & (df["destination_city"] == destination)]
    if not match.empty:
        dist = match["distance_miles"].mean()
        return round(dist, 2) if dist >= 25 else None
    return None

def fallback_haversine_distance(origin, destination):
    o = city_coords_map.get(origin)
    d = city_coords_map.get(destination)
    if o and d:
        miles = geodesic(o, d).miles
        return round(miles, 2) if miles >= 25 else None
    return None

# --- UI Code ---
def run_pricing_simulator():
    st.subheader("🚛 Dynamic Freight Pricing Estimator (Industry Aligned)")

    with st.form("pricing_form"):
        col1, col2 = st.columns(2)

        with col1:
            origin = st.selectbox("Origin City", [""] + valid_cities)
            destination = st.selectbox("Destination City", [""] + valid_cities)
            commodity = st.selectbox("Commodity", sorted(df["commodity_description"].dropna().unique()))
            weight_tons = st.number_input("Shipment Weight (tons)", min_value=0.1, value=10.0)

        with col2:
            fuel_price = st.number_input("Fuel Price ($/gal)", min_value=1.0, value=4.00)
            month = st.selectbox("Month of Shipment", list(range(1, 13)), index=2)
            truck_options = get_allowed_trucks(commodity)
            truck_type = st.selectbox("Truck Type", truck_options)
            is_rush = st.checkbox("Rush/Urgent Delivery?")
            guaranteed = st.checkbox("Guaranteed by Noon?")
            liftgate = st.checkbox("Liftgate Service?")
            inside = st.checkbox("Inside Delivery?")

        # --- Advanced tuning ---
        with st.expander("📐 Advanced Options & Benchmark Controls"):
            base_rate_override = st.number_input("Override Base Rate ($/mile)", min_value=0.0, value=0.0, step=0.01)
            fuel_baseline = st.number_input("Baseline Fuel Price for Surcharge", min_value=1.0, value=4.00, step=0.1)
            profit_margin = st.slider("Profit Margin (%)", 0, 100, value=20, step=5) / 100

        # --- Estimate button ---
        submitted = st.form_submit_button("Estimate Price")

        if submitted:
            if not origin or not destination:
                st.error("Please select both origin and destination.")
                return

            # Get distances
            avg_dist = get_avg_distance(df, origin, destination)
            fallback_dist = fallback_haversine_distance(origin, destination)
            final_dist = avg_dist or fallback_dist or 300.0

            if avg_dist:
                st.caption("📊 Using average from dataset.")
            elif fallback_dist:
                st.caption("🧭 Using geolocation estimate.")
            else:
                st.caption("⚠️ Using default fallback distance of 300 miles.")

            # Get class
            commodity_class = get_commodity_class(commodity)

            # Call pricing logic
            estimated_cost, breakdown = estimate_price_with_breakdown(
                weight_tons=weight_tons,
                distance_miles=final_dist,
                fuel_price=fuel_price,
                month=month,
                commodity_class=commodity_class,
                truck_type=truck_type,
                is_rush=is_rush,
                guaranteed_by_noon=guaranteed,
                liftgate_service=liftgate,
                inside_delivery=inside,
                base_rate_override=base_rate_override if base_rate_override > 0 else None,
                profit_margin=profit_margin,
                fuel_baseline=fuel_baseline
            )

            # Display main estimate
            st.success(
                f"💲 Estimated Cost: **${estimated_cost:,}** "
                f"for shipping {weight_tons} tons of {commodity} "
                f"from {origin} to {destination} ({final_dist} miles)."
            )

            # --- Cost Breakdown Section ---
            with st.expander("📊 Cost Breakdown & Methodology"):
                st.markdown(f"**Linehaul Cost:** ${breakdown['Linehaul Cost']:,}")
                st.markdown(f"**Fuel Surcharge:** ${breakdown['Fuel Surcharge']:,}")
                if breakdown["Accessorial Charges"]:
                    st.markdown("**Accessorial Charges:**")
                    for k, v in breakdown["Accessorial Charges"].items():
                        st.markdown(f"- {k}: ${v:,}")
                st.markdown(f"**Profit (Markup):** ${breakdown['Profit (Markup)']:,}")
                st.markdown(f"**Total Estimated Cost:** ${breakdown['Final Total']:,}")
                st.caption(f"Rate used: **${breakdown['Adjusted Rate ($/mile)']} per mile** "
                           f"(Base: ${breakdown['Base Rate Used']} | Class: {commodity_class})")

                st.markdown("---")
                st.markdown("### 📘 How This Is Calculated")
                st.markdown("""
                - **Linehaul** = distance × rate × class adjustment  
                - **Fuel surcharge** = % added if fuel exceeds baseline  
                - **Accessorials** = flat or % fees for added services  
                - **Markup** = added profit (e.g., 20%)  
                - **Minimum charge** = enforced ($150 default)
                """)

# Optional: for dev preview
if __name__ == "__main__":
    run_pricing_simulator()