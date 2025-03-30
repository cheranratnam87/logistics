
![logistics_gif](https://github.com/user-attachments/assets/e806ce72-3cb6-44b1-af45-5b7b03c10138)

🚚 Freight Pricing & Optimization Dashboard
This Streamlit app provides a dynamic freight cost estimator and a profit-maximizing shipment selector using FAF5 data. It's built to simulate real-world logistics pricing by incorporating factors like fuel prices, seasonality, shipment weight, and route characteristics.

🔍 What It Does
Freight Pricing Estimator
Input origin, destination, weight, month, and fuel price to get an estimated shipping cost. The pricing logic factors in:

FAF5 truck flow data (2022)

Seasonality

Fuel price multipliers

Commodity type and shipment weight

Minimum charge thresholds

Optimization Engine
Powered by Gurobi, the app includes an optimization module that:

Selects the most profitable subset of shipments

Respects constraints like truck capacity, total distance, fuel cost, and max number of shipments

Can be adapted for either shipment selection or route planning

🧱 Tech Stack
Python

Streamlit for the interactive dashboard

Gurobi for linear optimization

Pandas, NumPy for data wrangling

FAF5 Data (Freight Analysis Framework) — 2022 Truck Flow Dataset

📁 Project Structure
bash
Copy
Edit
├── app.py                 # Streamlit dashboard
├── pricing_engine.py      # Freight pricing logic
├── optimizer.py           # Gurobi optimization module
├── utils.py               # Helper functions
├── data/                  # Cleaned FAF5 and metadata files
└── README.md              # This file

💡 How to Run Locally
Clone the repo:

git clone https://github.com/yourusername/freight-pricing-app.git
cd freight-pricing-app

Install dependencies:

pip install -r requirements.txt
Add your Gurobi license if running optimization locally.

Run the app:

streamlit run app.py

🚀 Try It Live
Check out the live version here:

Streamlit App: https://logistics-kk34nzr4hekiwm2tpxhrmx.streamlit.app/

📌 Notes
Gurobi is used for optimization — you'll need a valid license to run locally. Free academic licenses are available at gurobi.com.

The pricing model is customizable for additional features like urgency, truck type, or congestion pricing.

Optimization logic can be extended to route planning and network flow models.

🧑‍💻 Author
Cheran Ratnam
