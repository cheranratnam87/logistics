import streamlit as st
import pandas as pd
import numpy as np
import math
import ast
import gurobipy as gp
from gurobipy import GRB
from geopy.distance import geodesic, distance
from geopy.geocoders import Nominatim
from functools import lru_cache
from scipy.spatial import KDTree

@lru_cache(maxsize=None)
def cached_geodesic(coord1, coord2):
    """Cache geodesic distance calculations for performance."""
    return geodesic(coord1, coord2).miles

@st.cache_data
def load_data(file_path="data/cleaned_freight_data.csv"):
    df = pd.read_csv(file_path)
    if 'price_per_ton_mile' not in df.columns:
        df['price_per_ton_mile'] = df['estimated_cost'] / (df['distance_miles'] * df['weight_tons'])
    return df

@st.cache_data
def load_cng_stations(file_path="data/all_stations_allfuelTypes.csv"):
    df = pd.read_csv(file_path)
    return df[['Station Name', 'Latitude', 'Longitude']].dropna()

def parse_coords(coord_str):
    try:
        return ast.literal_eval(coord_str)
    except:
        return None

def calculate_bearing(pointA, pointB):
    lat1, lat2 = math.radians(pointA[0]), math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def build_fuel_station_index(df_stations):
    """Build a KDTree for fast nearest-neighbor searches of fueling stations."""
    coords = df_stations[['Latitude', 'Longitude']].dropna().values
    return KDTree(coords), coords

def find_nearest_station(run_out_point, fuel_station_tree, df_stations):
    """Find the nearest fueling station to a given point."""
    dist, idx = fuel_station_tree.query(run_out_point)
    nearest_station = df_stations.iloc[idx]
    return nearest_station, dist

geolocator = Nominatim(user_agent="myGeocoder")

def run_optimization_tool(preloaded_data):
    st.header("📊 Gurobi Shipment Optimizer")
    df = preloaded_data
    sample_size = st.slider("Number of shipments to simulate", 50, 1000, 300, 50)
    max_shipments = st.number_input("Max number of shipments", 1, 1000, 300)
    max_tons = st.number_input("Max total tons", 10, 1_000_000, 200_000)
    max_distance = st.number_input("Max total miles", 10, 1_000_000, 50_000)
    st.slider("Fuel price ($/gallon)", 2.0, 6.0, 4.0, 0.1)
    st.selectbox("Month", list(range(1, 13)))

    if st.button("Run Optimization"):
        df_sample = df.sample(min(sample_size, len(df)), random_state=42).copy()
        df_sample['tons'] = np.random.randint(5, 40, len(df_sample))
        df_sample['commodity_code'] = np.random.choice(range(1, 44), len(df_sample))

        def calculate_profit(row):
            cost = row['tons'] * row['distance_miles'] * row['price_per_ton_mile']
            adj = cost * 1.1
            return adj * 1.2, adj * 0.2

        df_sample['profit'] = df_sample.apply(lambda row: calculate_profit(row)[1], axis=1)

        model = gp.Model("shipment_opt")
        model.setParam("OutputFlag", 0)
        x = model.addVars(len(df_sample), vtype=GRB.BINARY)

        model.setObjective(gp.quicksum(df_sample['profit'].iloc[i] * x[i] for i in range(len(df_sample))), GRB.MAXIMIZE)
        model.addConstr(gp.quicksum(df_sample['tons'].iloc[i] * x[i] for i in range(len(df_sample))) <= max_tons)
        model.addConstr(gp.quicksum(df_sample['distance_miles'].iloc[i] * x[i] for i in range(len(df_sample))) <= max_distance)
        model.addConstr(gp.quicksum(x[i] for i in range(len(df_sample))) <= max_shipments)
        model.optimize()

        selected = [i for i in range(len(df_sample)) if x[i].X > 0.5]
        result = df_sample.iloc[selected]

        st.subheader("🚛 Optimization Results")
        st.write(result[['origin_city', 'destination_city', 'tons', 'distance_miles', 'profit']])
        st.success(f"Total Profit: ${result['profit'].sum():,.2f} | Tons: {result['tons'].sum():,.0f} | Distance: {result['distance_miles'].sum():,.0f} mi")

def run_multi_destination_route_optimizer(df, df_stations):
    st.header("🗺️ Route Optimizer with Refueling Logic")
    df['origin_coords'] = df['origin_coords'].apply(parse_coords)
    df['destination_coords'] = df['destination_coords'].apply(parse_coords)

    all_cities = sorted(set(df["origin_city"]).union(df["destination_city"]))
    origin = st.selectbox("Origin City", all_cities, index=all_cities.index("Atlanta GA"))
    destinations = st.multiselect("Destination Cities", [c for c in all_cities if c != origin])
    truck_range = st.number_input("Truck Fuel Range (mi)", 100, 2000, 500)

    if not destinations:
        st.warning("Please select at least one destination.")
        return

    if st.button("Optimize Route"):
        def get_coords(city):
            c = df[df['origin_city'] == city]['origin_coords'].dropna().values
            if not len(c): c = df[df['destination_city'] == city]['destination_coords'].dropna().values
            if len(c): return ast.literal_eval(c[0])
            loc = geolocator.geocode(city)
            return (loc.latitude, loc.longitude) if loc else None

        cities = [origin] + destinations
        coords_dict = {c: get_coords(c) for c in cities}
        if None in coords_dict.values():
            st.error("Missing coords for some cities.")
            return

        n = len(cities)
        dist = {(i, j): cached_geodesic(coords_dict[cities[i]], coords_dict[cities[j]]) for i in range(n) for j in range(n) if i != j}
        model = gp.Model("route_optimizer")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 60  # Set a time limit for optimization
        model.Params.MIPGap = 0.01  # Allow a small optimality gap

        x = model.addVars(dist.keys(), vtype=GRB.BINARY)
        u = model.addVars(n, lb=0, ub=n - 1)

        model.setObjective(gp.quicksum(dist[i, j] * x[i, j] for i, j in dist), GRB.MINIMIZE)
        model.addConstrs((x.sum(i, '*') == 1 for i in range(n)), name="outflow")
        model.addConstrs((x.sum('*', i) == 1 for i in range(n)), name="inflow")
        model.addConstrs((u[i] - u[j] + n * x[i, j] <= n - 1 for i in range(1, n) for j in range(1, n) if i != j), name="subtour")
        model.optimize()

        route = [0]
        visited = {0}
        while len(route) < n:
            for j in range(n):
                if j not in visited and x[route[-1], j].X > 0.5:
                    route.append(j)
                    visited.add(j)
                    break

        ordered = [cities[i] for i in route]
        details = {}
        total_distance = 0

        # Build KDTree for fuel stations
        fuel_station_tree, fuel_station_coords = build_fuel_station_index(df_stations)

        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            coord_a, coord_b = coords_dict[a], coords_dict[b]
            leg_dist = cached_geodesic(coord_a, coord_b)
            total_distance += leg_dist
            fuel_stops = []
            current_pos = coord_a
            remaining = leg_dist

            while remaining > truck_range:
                run_out = distance(miles=truck_range).destination(current_pos, calculate_bearing(current_pos, coord_b))
                run_out_point = (run_out.latitude, run_out.longitude)
                nearest_station, min_dist = find_nearest_station(run_out_point, fuel_station_tree, df_stations)
                if nearest_station is None:
                    break
                fuel_stops.append(nearest_station['Station Name'])
                current_pos = (nearest_station['Latitude'], nearest_station['Longitude'])
                remaining = cached_geodesic(current_pos, coord_b)

            details[f"{a} ➡ {b}"] = {
                "leg_distance": leg_dist,
                "fuel_stops": fuel_stops,
                "final_leg": remaining
            }

        st.subheader("Optimized Visit Order:")
        st.write(ordered)
        st.success(f"🧭 Total Route Distance: {total_distance:.2f} mi")
        st.markdown("### ⛽ Fueling Stop Details")
        st.write(details)

def main():
    st.set_page_config("Freight Optimizer", layout="wide")
    st.sidebar.title("Navigation")
    tool = st.sidebar.selectbox("Choose Tool", ["Shipment Optimizer", "Route Optimizer"])
    df = load_data()
    df_stations = load_cng_stations()
    if tool == "Shipment Optimizer":
        run_optimization_tool(df)
    else:
        run_multi_destination_route_optimizer(df, df_stations)

if __name__ == "__main__":
    main()