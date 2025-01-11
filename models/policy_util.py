def apply_trend_adjustment(base_value, trend_factor):
    return int(base_value * trend_factor)

def pricing_decision(base_price, competitor_price, trend_factor=1.0):
    price = max(0, base_price - 50 * trend_factor)
    discount = max(0, competitor_price - price)
    return price, discount

def basic_order_up_to_policy_util(inventory, theta_min, theta_max, demand_forecast, trend_factors, variance_demand):
    restock = {}
    for car, level in inventory.items():
        safety_buffer = int(variance_demand[car] ** 0.5)
        restock[car] = max(0, min(
            theta_max[car] - level,
            trend_factors[car] * (demand_forecast[car] + safety_buffer)
        ))
    return restock
