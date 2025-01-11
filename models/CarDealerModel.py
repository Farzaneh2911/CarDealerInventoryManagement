from collections import namedtuple
import numpy as np
from BaseClasses.SDPModel import SDPModel
import pandas as pd

State = namedtuple("State", ["inventory_level", "holding_time", "competitor_price", "market_trends","market_trends_lag1",
    "market_trends_lag2", "demand_forecast","trend_eta_params"])
Decision = namedtuple("Decision", ["restock", "price", "discount"])

class CarDealerInventoryModel(SDPModel):
    """
    Comprehensive Car Dealer Inventory Management Model.
    """

    def __init__(self, S0, T=30, seed=42, holding_cost=85, base_profit=1500, discount_rate=0.05, max_inventory=100,
                 seasonal_holding_cost=None, alpha=0.1, market_trends_file=None):
        """
        Initializes the Car Dealer Model.
        """
        state_names = ["inventory_level", "holding_time", "competitor_price", "market_trends", "demand_forecast"]
        decision_names = ["restock", "price", "discount"]

        super().__init__(state_names, decision_names, S0, T=T, seed=seed)
        self.holding_cost = holding_cost
        self.base_profit = base_profit
        self.discount_rate = discount_rate
        self.max_inventory = max_inventory
        self.alpha = alpha
        self.variance_demand = {car: 0 for car in S0["inventory_level"]}
        self.variance_forecast = {car: 0 for car in S0["inventory_level"]}
        self.seasonal_holding_cost = seasonal_holding_cost or {"winter": 90, "summer": 70}

        # Load market trends if provided
        self.market_trends_df = pd.read_csv(market_trends_file) if market_trends_file else None

        self.state = State(
            inventory_level=S0["inventory_level"],
            holding_time=S0["holding_time"],
            competitor_price=S0["competitor_price"],
            market_trends=S0.get("market_trends", {car: "stable" for car in S0["inventory_level"]}),
            market_trends_lag1=S0.get("market_trends_lag1", {car: "stable" for car in S0["inventory_level"]}),
            market_trends_lag2=S0.get("market_trends_lag2", {car: "stable" for car in S0["inventory_level"]}),
            demand_forecast=S0["demand_forecast"],
            trend_eta_params={"stable": [0.6, 0.3, 0.1], "rising": [0.7, 0.2, 0.1], "declining": [0.5, 0.3, 0.2]}
        )

    def exog_info_fn(self, decision):
        """
        Generate exogenous information using market trends.
        """
        # Determine the current month
        current_month = f"2023-{(self.t % 12) + 1:02d}"  # Assume year 2023 for simplicity
        trends_for_month = self.market_trends_df[self.market_trends_df["Month"] == current_month]

        # Update market trends and calculate demand
        demand = {}
        competitor_prices = {}
        new_market_trends = {}

        for car in self.state.inventory_level:
            car_trend_data = trends_for_month[trends_for_month["Model"] == car]
            if not car_trend_data.empty:
                demand_index = car_trend_data["Demand_Index"].values[0]
                price_trend = car_trend_data["Price_Trend"].values[0]
                competitor_activity = car_trend_data["Competitor_Activity"].values[0]

                # Use market trends to influence demand
                base_demand = self.state.demand_forecast.get(car, 0)
                random_variation = self.prng.normal(0, 0.05)
                demand = {
                 car: max(0, self.prng.normal(self.state.demand_forecast[car], 10))
            for car in self.state.inventory_level
        }
                competitor_prices[car] = max(0, self.state.competitor_price[car] + price_trend - competitor_activity)
                new_market_trends[car] = car_trend_data["Season"].values[0]
            else:
                # Fallback to default values
                 base_demand = self.state.demand_forecast.get(car, 0)
                 demand[car] = max(
                  0,
                  base_demand + self.prng.normal(0, 5)  # Add random fluctuation to base demand
                )
                 competitor_prices[car] = self.state.competitor_price[car]
                 new_market_trends[car] = "stable"
        max_demand = {car: 2 * demand for car, demand in self.state.demand_forecast.items()}  # Cap at twice the forecasted demand
 # Cap at twice the forecasted demand
        demand = {car: min(demand[car], max_demand.get(car, demand[car])) for car in demand}         

        return {
            "demand": demand,
            "competitor_prices": competitor_prices,
            "market_trends": new_market_trends,
        }

    def transition_fn(self, decision, exog_info):
        """
        Updates state variables based on exogenous information.
        """
        new_inventory = {}
        new_holding_time = {}
        cars_sold = {}
        cars_added = {}

        for car in self.state.inventory_level:
            # Calculate cars sold, capped by available inventory and demand
            demand = int(round(exog_info.get("demand", {}).get(car, 0)))
            cars_sold[car] = min(demand, self.state.inventory_level[car])
            # Cars added (restocked)
            cars_added[car] = int(decision.restock.get(car, 0))
            # Update inventory
            new_inventory[car] = max(0, self.state.inventory_level[car] + cars_added[car] - cars_sold[car])

        # Update holding times
        new_holding_time = {
            car: self.state.holding_time[car] + 1 if new_inventory[car] > 0 else 0
            for car in self.state.inventory_level
        }
        

        # Shift market trends
        market_trends_lag2 = self.state.market_trends_lag1
        market_trends_lag1 = self.state.market_trends
        market_trends = exog_info["market_trends"]
        new_demand_forecast = {
           car: max(0, exog_info.get("demand", {}).get(car, 0) * (1 + 0.1 * self.t))
           for car in self.state.demand_forecast
        }

        return {
            "inventory_level": new_inventory,
            "holding_time": new_holding_time,
            "competitor_price": exog_info["competitor_prices"],
            "market_trends": market_trends,
            "market_trends_lag1": market_trends_lag1,
            "market_trends_lag2": market_trends_lag2,
            "demand_forecast": new_demand_forecast,
            "trend_eta_params": self.state.trend_eta_params,
            "cars_sold": cars_sold,
            "cars_added": cars_added,
        }


    def objective_fn(self, decision, exog_info):
        """
        Calculates the profit for the current step.
        """
        fixed_cost_per_unit = 300 
        revenue = sum(min(self.state.inventory_level[car], exog_info["demand"][car]) * decision.price[car]
                      for car in self.state.inventory_level)
        restocking_cost = sum(decision.restock[car] * fixed_cost_per_unit
                              for car in decision.restock)
        unsold_inventory = {car: self.state.inventory_level[car] - min(self.state.inventory_level[car], int(exog_info["demand"][car]))
                    for car in self.state.inventory_level}

        
        
         # Seasonal holding cost
        holding_cost = sum(
          unsold_inventory[car] * self.state.holding_time[car] * 
          self.seasonal_holding_cost.get(exog_info["market_trends"][car].lower(), self.holding_cost)
          for car in unsold_inventory
    )

        

         
        profit = revenue - (restocking_cost + holding_cost)
        
        return profit
        


    def is_valid_decision(self, decision):
        """
        Validates decision.
        """
        total_inventory = sum(self.state.inventory_level[car] + decision.restock[car] for car in self.state.inventory_level)
        if any(value < 0 for value in decision.restock.values()):
         raise ValueError("Restocking values cannot be negative!")
        if total_inventory > self.max_inventory:
            print(f"Decision Invalid: Restocking {decision.restock} would exceed max inventory of {self.max_inventory}.")
            raise ValueError("Total inventory exceeds storage capacity!")
        return True