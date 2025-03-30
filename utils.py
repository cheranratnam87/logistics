def get_commodity_class(commodity):
    """
    Returns the commodity class based on the commodity description.
    For simplicity, this function assigns a fixed class to each commodity.
    """
    commodity_classes = {
        "Alcoholic beverages": 70,
        "Electronics": 85,
        "Furniture": 100,
        "Machinery": 92.5,
        "Food": 65,
        "Clothing": 77.5,
    }
    return commodity_classes.get(commodity, 50)  # Default to class 50 if not found


def get_allowed_trucks(commodity):
    """
    Returns a list of allowed truck types based on the commodity.
    """
    if commodity in ["Alcoholic beverages", "Food"]:
        return ["Dry Van", "Reefer"]
    elif commodity in ["Electronics", "Furniture"]:
        return ["Dry Van", "Flatbed"]
    elif commodity == "Machinery":
        return ["Flatbed"]
    else:
        return ["Dry Van"]


def estimate_price_with_breakdown(weight_tons, distance_miles, fuel_price, month,
                                  commodity_class, truck_type, is_rush,
                                  guaranteed_by_noon, liftgate_service,
                                  inside_delivery, base_rate_override=None,
                                  profit_margin=0.2, fuel_baseline=4.0):
    """
    Estimates the price of freight shipping and provides a detailed cost breakdown.
    """
    # Default base rate per ton-mile
    base_rate_per_ton_mile = base_rate_override if base_rate_override else 2.0

    # Seasonal adjustment factors for each month
    seasonal_adjustments = {
        1: 1.05,  # January (5% increase)
        2: 1.03,  # February (3% increase)
        3: 1.00,  # March (no change)
        4: 0.97,  # April (3% decrease)
        5: 0.95,  # May (5% decrease)
        6: 1.00,  # June (no change)
        7: 1.10,  # July (10% increase)
        8: 1.08,  # August (8% increase)
        9: 1.00,  # September (no change)
        10: 0.98,  # October (2% decrease)
        11: 1.02,  # November (2% increase)
        12: 1.07,  # December (7% increase)
    }

    # Truck type cost multipliers
    truck_type_multipliers = {
        "Dry Van": 1.0,  # No additional cost
        "Reefer": 1.15,  # 15% increase for refrigerated trucks
        "Flatbed": 1.10,  # 10% increase for flatbed trucks
    }

    # Get the seasonal adjustment factor for the selected month
    seasonal_factor = seasonal_adjustments.get(month, 1.0)

    # Get the truck type multiplier
    truck_multiplier = truck_type_multipliers.get(truck_type, 1.0)

    # Calculate linehaul cost (weight × distance × rate)
    linehaul_cost = weight_tons * distance_miles * base_rate_per_ton_mile

    # Apply commodity class adjustment
    class_adjustment = 1 + (commodity_class - 50) / 100
    linehaul_cost *= class_adjustment

    # Apply seasonal adjustment
    linehaul_cost *= seasonal_factor

    # Apply truck type multiplier
    linehaul_cost *= truck_multiplier

    # Apply rush delivery multiplier
    rush_multiplier = 1.25 if is_rush else 1.0
    linehaul_cost *= rush_multiplier

    # Calculate fuel surcharge
    fuel_surcharge = 0
    if fuel_price > fuel_baseline:
        fuel_surcharge = linehaul_cost * ((fuel_price - fuel_baseline) / fuel_baseline)

    # Calculate accessorial charges
    accessorial_charges = {}
    if guaranteed_by_noon:
        accessorial_charges["Guaranteed by Noon"] = 50
    if liftgate_service:
        accessorial_charges["Liftgate Service"] = 25
    if inside_delivery:
        accessorial_charges["Inside Delivery"] = 30

    total_accessorials = sum(accessorial_charges.values())

    # Calculate profit
    profit = (linehaul_cost + fuel_surcharge + total_accessorials) * profit_margin

    # Final total cost
    final_total = linehaul_cost + fuel_surcharge + total_accessorials + profit

    # Return breakdown
    return final_total, {
        "Linehaul Cost": linehaul_cost,
        "Fuel Surcharge": fuel_surcharge,
        "Accessorial Charges": accessorial_charges,
        "Profit (Markup)": profit,
        "Final Total": final_total,
        "Adjusted Rate ($/mile)": base_rate_per_ton_mile,
        "Base Rate Used": base_rate_per_ton_mile
    }