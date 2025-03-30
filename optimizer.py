import streamlit as st
import pandas as pd
import numpy as np
import math
import ast
import gurobipy as gp
from gurobipy import GRB
from geopy.distance import geodesic, distance
from geopy.geocoders import Nominatim

# --------------------------------
# Shared Helper Functions & Data Loading
# --------------------------------

@st.cache_data
def load_data(file_path="data/cleaned_freight_data.csv"):
    df = pd.read_csv(file_path)
    # Ensure 'price_per_ton_mile' is calculated if missing
    if 'price_per_ton_mile' not in df.columns:
        df['price_per_ton_mile'] = df['estimated_cost'] / (df['distance_miles'] * df['weight_tons'])
    return df

@st.cache_data
def load_cng_stations(file_path="data/all_stations_allfuelTypes.csv"):
    """Load all fueling stations from the updated dataset."""
    df = pd.read_csv(file_path)
    # Ensure the dataset has necessary columns
    required_columns = ['Station Name', 'Latitude', 'Longitude']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The fueling station dataset must contain the following columns: {required_columns}")
    return df

def parse_coords(coord_str):
    try:
        return ast.literal_eval(coord_str)
    except Exception as e:
        return None

def calculate_bearing(pointA, pointB):
    """Calculate the compass bearing from pointA to pointB."""
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    return (initial_bearing + 360) % 360

# Initialize geocoder (only once)
geolocator = Nominatim(user_agent="myGeocoder")

# --------------------------------
# Shipment Optimizer (Existing Code)
# --------------------------------

def run_optimization_tool():
    st.header("📊 Gurobi Shipment Optimizer")

    st.markdown("""
    ### 🚚 How the Shipment Optimizer Works
    The **Shipment Optimizer** selects the most profitable shipments based on constraints like weight, distance, and fuel costs. It uses **Gurobi** to solve the optimization problem, selecting shipments that maximize profit while staying within the specified limits.
    
    **Input Parameters:**
    - **Number of Shipments to Simulate**: The number of shipments the optimizer randomly selects for evaluation.
    - **Max Number of Shipments**: The upper limit on the number of shipments the optimizer can select.
    - **Max Total Tons**: The maximum weight of selected shipments.
    - **Max Total Miles**: The maximum distance the selected shipments will cover.
    - **Fuel Price**: The price of fuel, which is used to calculate fuel surcharges.
    - **Month**: Adjustments based on seasonality or fluctuations in demand.
    
    **Optimization Process:**
    - The optimizer evaluates possible combinations of shipments that meet these constraints and selects the set that maximizes the total profit.
    """)

    uploaded_file = st.file_uploader("Upload your freight data file (CSV format)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.info("ℹ️ Using default dataset.")
        df = load_data()

    st.markdown("### 🚚 Simulation Settings")
    sample_size = st.slider("Number of shipments to simulate", 50, 1000, 300, step=50)
    max_shipments = st.number_input("Max number of shipments", 1, 1000, 300)
    max_tons = st.number_input("Max total tons", 10, 1_000_000, 200_000)
    max_distance = st.number_input("Max total miles", 10, 1_000_000, 50_000)
    fuel_price = st.slider("Fuel price ($/gallon)", 2.0, 6.0, 4.0, step=0.1)
    month = st.selectbox("Month", list(range(1, 13)))

    if st.button("Run Optimization", key="optimizer_run_button"):
        if sample_size > len(df):
            st.warning(f"⚠️ The number of shipments to simulate ({sample_size}) exceeds the available data ({len(df)} rows). Adjusting to use all available data.")
            sample_size = len(df)

        df_sample = df.sample(n=sample_size, random_state=42).copy()
        df_sample['tons'] = np.random.randint(5, 40, size=len(df_sample))
        df_sample['commodity_code'] = np.random.choice(list(range(1, 44)), size=len(df_sample))

        def calculate_profit(row):
            base_cost = row['tons'] * row['distance_miles'] * row['price_per_ton_mile']
            adjusted_cost = base_cost * 1.1
            price = adjusted_cost * 1.2
            return price, price - adjusted_cost

        df_sample[['price', 'profit']] = df_sample.apply(lambda row: pd.Series(calculate_profit(row)), axis=1)

        model = gp.Model("shipment_optimization")
        model.setParam("OutputFlag", 0)
        x = model.addVars(len(df_sample), vtype=GRB.BINARY, name="x")

        model.setObjective(gp.quicksum(df_sample['profit'].iloc[i] * x[i] for i in range(len(df_sample))), GRB.MAXIMIZE)
        model.addConstr(gp.quicksum(df_sample['tons'].iloc[i] * x[i] for i in range(len(df_sample))) <= max_tons)
        model.addConstr(gp.quicksum(df_sample['distance_miles'].iloc[i] * x[i] for i in range(len(df_sample))) <= max_distance)
        model.addConstr(gp.quicksum(x[i] for i in range(len(df_sample))) <= max_shipments)
        model.optimize()

        selected_indices = [i for i in range(len(df_sample)) if x[i].X > 0.5]
        selected_df = df_sample.iloc[selected_indices]

        total_profit = selected_df['profit'].sum()
        total_tons = selected_df['tons'].sum()
        total_distance = selected_df['distance_miles'].sum()

        st.markdown("### 🚛 Optimization Results")
        st.success(f"✅ Selected {len(selected_df)} shipments")
        st.write(selected_df[['origin_city', 'destination_city', 'tons', 'distance_miles', 'profit']])
        st.markdown(f"""
        **Total Profit**: ${total_profit:,.2f}  
        **Total Tons**: {total_tons:,.2f} tons  
        **Total Distance**: {total_distance:,.2f} miles
        """)

# --------------------------------
# Single-Destination Route Optimizer
# --------------------------------

def run_route_optimizer_ui():
    st.header("🗺️ Route Optimizer with Fuel Planning")
    
    df_freight = load_data()
    df_stations = load_cng_stations()  # Use the updated fueling station dataset
    
    # Convert coordinate strings to tuples
    df_freight['origin_coords'] = df_freight['origin_coords'].apply(parse_coords)
    df_freight['destination_coords'] = df_freight['destination_coords'].apply(parse_coords)
    
    all_cities = sorted(set(df_freight["origin_city"]).union(df_freight["destination_city"]))
    
    # Default origin and destinations
    default_origin = "Atlanta GA"
    default_destinations = ["Austin TX", "Birmingham AL"]
    
    origin = st.selectbox("Select Origin City", all_cities, index=all_cities.index(default_origin), key="route_origin")
    destinations = st.multiselect(
        "Select Destination Cities (Max: 2)", 
        [c for c in all_cities if c != origin], 
        default=default_destinations, 
        key="route_destinations", 
        max_selections=2  # Enforce a maximum of two selections
    )
    truck_range = st.number_input("Truck Fuel Range (miles per tank)", min_value=100, max_value=2000, value=500, key="route_range")
    
    if st.button("Plan Route", key="plan_route_button"):
        if not destinations:
            st.error("🚫 Please select at least one destination.")
            return
        
        st.success(f"🚛 Planning route from **{origin}** to destinations: {', '.join(destinations)}")
        st.info(f"🔋 Max range before refueling: **{truck_range} miles**")
        
        def get_coords(city):
            coords = df_freight.loc[df_freight['origin_city'] == city, 'origin_coords'].values
            if len(coords) == 0:
                coords = df_freight.loc[df_freight['destination_city'] == city, 'destination_coords'].values
            if len(coords) > 0 and coords[0]:
                return coords[0]
            location = geolocator.geocode(city)
            if location:
                return (location.latitude, location.longitude)
            return None

        # Get coordinates for origin and destinations
        coords_dict = {origin: get_coords(origin)}
        for dest in destinations:
            coords_dict[dest] = get_coords(dest)
        
        if None in coords_dict.values():
            st.error("🚫 Could not find coordinates for one or more selected cities.")
            return
        
        # Build distance matrix (geodesic distances)
        required_nodes = [origin] + destinations
        n = len(required_nodes)
        d = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    d[(i, j)] = 0
                else:
                    d[(i, j)] = geodesic(coords_dict[required_nodes[i]], coords_dict[required_nodes[j]]).miles
        
        # Solve Hamiltonian Path (TSP variant) using Gurobi
        model = gp.Model("HamiltonianPath")
        model.Params.OutputFlag = 0
        
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        model.update()
        
        # Constraints
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, f"out_{i}")
            model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1, f"in_{i}")
        
        # Objective: Minimize total distance
        model.setObjective(gp.quicksum(d[(i, j)] * x[i, j] for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)
        model.optimize()
        
        # Extract route order starting from origin (index 0)
        route_order = [0]
        current = 0
        while len(route_order) < n:
            for j in range(n):
                if j != current and x[current, j].X > 0.5:
                    route_order.append(j)
                    current = j
                    break
        
        optimized_route = [required_nodes[i] for i in route_order]
        st.write("### Optimized Visit Order:")
        st.write(optimized_route)
        
        # Insert fueling stops for the entire route
        total_route_distance = 0
        full_route_details = {}
        for idx in range(len(optimized_route) - 1):
            leg_origin = optimized_route[idx]
            leg_destination = optimized_route[idx + 1]
            leg_origin_coord = coords_dict[leg_origin]
            leg_destination_coord = coords_dict[leg_destination]
            leg_distance = geodesic(leg_origin_coord, leg_destination_coord).miles
            total_route_distance += leg_distance
            
            st.write(f"**Leg:** {leg_origin} ➡ {leg_destination} | Distance: {leg_distance:.2f} miles")
            current_pos = leg_origin_coord
            leg_fuel_stops = []
            remaining_leg_distance = geodesic(current_pos, leg_destination_coord).miles
            while remaining_leg_distance > truck_range:
                run_out_obj = distance(miles=truck_range).destination(current_pos, calculate_bearing(current_pos, leg_destination_coord))
                run_out_point = (run_out_obj.latitude, run_out_obj.longitude)
                min_dist = float('inf')
                nearest_station = None
                for _, row in df_stations.iterrows():
                    if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
                        continue  # Skip stations with missing coordinates
                    station_coord = (row['Latitude'], row['Longitude'])
                    d_station = geodesic(run_out_point, station_coord).miles
                    if d_station < min_dist:
                        min_dist = d_station
                        nearest_station = row['Station Name']
                if nearest_station is None:
                    st.warning(f"⚠️ No fueling station found near the projected run-out point for leg {leg_origin} ➡ {leg_destination}.")
                    break  # Exit the loop if no station is found
                leg_fuel_stops.append(nearest_station)
                st.write(f"⛽ Refueling stop on leg {leg_origin} ➡ {leg_destination}: **{nearest_station}** (≈{min_dist:.2f} miles off run-out point)")
                station_row = df_stations[df_stations['Station Name'] == nearest_station].iloc[0]
                current_pos = (station_row['Latitude'], station_row['Longitude'])
                remaining_leg_distance = geodesic(current_pos, leg_destination_coord).miles
            
            full_route_details[f"{leg_origin} ➡ {leg_destination}"] = {
                "leg_distance": leg_distance,
                "fuel_stops": leg_fuel_stops,
                "final_leg": remaining_leg_distance
            }
        
        st.write(f"**Total Route Distance (required nodes only):** {total_route_distance:.2f} miles")
        st.write("### Fueling Stop Details per Leg:")
        st.write(full_route_details)

        # Add a quick note about the calculation
        st.markdown("""
        ---
        ### How This is Calculated
        - **Route Optimization**: The route is optimized using the Gurobi solver to minimize the total distance while ensuring that each destination is visited exactly once.
        - **Constraints**:
          - Each city has exactly one incoming and one outgoing route.
          - Subtour elimination constraints are applied to prevent disconnected loops.
        - **Fuel Stops**: Fuel stops are calculated based on the truck's range and the geodesic distance between cities. The nearest fueling station to the projected run-out point is selected.
        """)

# --------------------------------
# Multi-Destination Route Optimizer (Using Gurobi)
# --------------------------------
def run_multi_destination_route_optimizer():
    st.header("🚚 Multi-Destination Route Optimizer with Fuel Planning")

    df_freight = load_data()
    df_stations = load_cng_stations()

    df_freight['origin_coords'] = df_freight['origin_coords'].apply(parse_coords)
    df_freight['destination_coords'] = df_freight['destination_coords'].apply(parse_coords)

    all_cities = sorted(set(df_freight["origin_city"]).union(df_freight["destination_city"]))

    # Default origin and destinations
    default_origin = "Atlanta GA"
    default_destinations = ["Columbus OH", "Beaumont TX"]

    origin = st.selectbox("Select Origin City", all_cities, index=all_cities.index(default_origin), key="multi_origin")
    destinations = st.multiselect(
        "Select Destination Cities (Max: 2)", 
        [c for c in all_cities if c != origin], 
        default=default_destinations, 
        key="multi_destinations"
    )
    truck_range = st.number_input("Truck Fuel Range (miles per tank)", min_value=100, max_value=2000, value=500, key="multi_truck_range")

    # Validate the number of destinations
    if len(destinations) > 2:
        st.error("🚫 You can select a maximum of two destinations.")
        return

    if st.button("Optimize Multi-Destination Route", key="multi_route_button"):
        if not destinations:
            st.error("Please select at least one destination.")
            return

        def get_coords(city):
            coords = df_freight.loc[df_freight['origin_city'] == city, 'origin_coords'].values
            if len(coords) == 0:
                coords = df_freight.loc[df_freight['destination_city'] == city, 'destination_coords'].values
            if len(coords) > 0 and coords[0]:
                return coords[0]
            location = geolocator.geocode(city)
            if location:
                return (location.latitude, location.longitude)
            return None

        coords_dict = {origin: get_coords(origin)}
        for dest in destinations:
            coords_dict[dest] = get_coords(dest)

        if None in coords_dict.values():
            st.error("🚫 Could not find coordinates for one or more selected cities.")
            return

        required_nodes = [origin] + destinations
        n = len(required_nodes)
        d = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    d[(i, j)] = 0
                else:
                    d[(i, j)] = geodesic(coords_dict[required_nodes[i]], coords_dict[required_nodes[j]]).miles

        model = gp.Model("HamiltonianPath")
        model.Params.OutputFlag = 0

        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # Add MTZ variables for subtour elimination
        u = {}
        for i in range(1, n):
            u[i] = model.addVar(lb=1, ub=n-1, vtype=GRB.INTEGER, name=f"u_{i}")

        model.update()

        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)
            model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)

        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)

        model.setObjective(gp.quicksum(d[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)
        model.optimize()

        # Extract optimized route
        def extract_mtz_route(x, n):
            route = []
            current = 0
            visited = set()
            while len(route) < n:
                route.append(current)
                visited.add(current)
                for j in range(n):
                    if j != current and (current, j) in x and x[current, j].X > 0.5 and j not in visited:
                        current = j
                        break
            return route

        route_order = extract_mtz_route(x, n)
        optimized_route = [required_nodes[i] for i in route_order]
        st.write("### Optimized Visit Order:")
        st.write(optimized_route)

        total_route_distance = 0
        full_route_details = {}

        for idx in range(len(optimized_route) - 1):
            leg_origin = optimized_route[idx]
            leg_destination = optimized_route[idx + 1]
            leg_origin_coord = coords_dict[leg_origin]
            leg_destination_coord = coords_dict[leg_destination]
            leg_distance = geodesic(leg_origin_coord, leg_destination_coord).miles
            total_route_distance += leg_distance

            st.write(f"**Leg:** {leg_origin} ➡ {leg_destination} | Distance: {leg_distance:.2f} miles")
            current_pos = leg_origin_coord
            leg_fuel_stops = []
            remaining_leg_distance = geodesic(current_pos, leg_destination_coord).miles
            while remaining_leg_distance > truck_range:
                run_out_obj = distance(miles=truck_range).destination(current_pos, calculate_bearing(current_pos, leg_destination_coord))
                run_out_point = (run_out_obj.latitude, run_out_obj.longitude)
                min_dist = float('inf')
                nearest_station = None
                for _, row in df_stations.iterrows():
                    station_coord = (row['Latitude'], row['Longitude'])
                    d_station = geodesic(run_out_point, station_coord).miles
                    if d_station < min_dist:
                        min_dist = d_station
                        nearest_station = row['Station Name']
                if nearest_station is None:
                    st.error("🚫 No fueling station found near the projected run-out point. Route cannot be completed.")
                    return
                leg_fuel_stops.append(nearest_station)
                st.write(f"⛽ Refueling stop on leg {leg_origin} ➡ {leg_destination}: **{nearest_station}** (≈{min_dist:.2f} miles off run-out point)")
                station_row = df_stations[df_stations['Station Name'] == nearest_station].iloc[0]
                current_pos = (station_row['Latitude'], station_row['Longitude'])
                remaining_leg_distance = geodesic(current_pos, leg_destination_coord).miles

            full_route_details[f"{leg_origin} ➡ {leg_destination}"] = {
                "leg_distance": leg_distance,
                "fuel_stops": leg_fuel_stops,
                "final_leg": remaining_leg_distance
            }

        st.write(f"**Total Route Distance (required nodes only):** {total_route_distance:.2f} miles")
        st.write("### Fueling Stop Details per Leg:")
        st.write(full_route_details)


# --------------------------------
# Main App Navigation
# --------------------------------

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the tool", 
                                    ["Shipment Optimizer", "Route Optimizer", "Multi-Destination Route Optimizer"])
    if app_mode == "Shipment Optimizer":
        run_optimization_tool()
    elif app_mode == "Route Optimizer":
        run_route_optimizer_ui()
    elif app_mode == "Multi-Destination Route Optimizer":
        run_multi_destination_route_optimizer()

if __name__ == "__main__":
    main()
