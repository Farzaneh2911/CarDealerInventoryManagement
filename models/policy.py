from collections import namedtuple
from copy import deepcopy
from CarDealerModel import State

# Define the Decision namedtuple
Decision = namedtuple("Decision", ["restock", "price", "discount"])

class CarDealerPolicy:
    def __init__(self, model, policy_name="OrderUpToPolicy"):
        self.model = model
        self.policy_name = policy_name

    def basic_order_up_to_policy(self, theta_min, theta_max, exog_info=None):
        """
        Implements a simple order-up-to policy:
        - Orders to bring inventory up to theta_max if below theta_min.

        Args:
            theta_min: Minimum inventory level to trigger restocking.
            theta_max: Target inventory level to restock up to.
            exog_info: Exogenous information (demand, trends, etc.).

        Returns:
            Decision object with restocking quantities.
        """
        restock = {}
        trend_adjustments = {"rising": 1.2, "stable": 1.0, "declining": 0.8}

        # Iterate through cars in the inventory
        for car, level in self.model.state.inventory_level.items():
            # Retrieve market trend and trend factor
            current_trend = self.model.state.market_trends.get(car, "stable")
            trend_factor = trend_adjustments.get(current_trend, 1.0)

            # Calculate demand forecast adjustment
            safety_buffer = int(self.model.variance_demand[car] ** 0.5)
            target_demand = exog_info["demand"][car] + safety_buffer

            # Compute restocking quantity
            restock[car] = max(0, min(
                theta_max[car] - level,
                trend_factor * target_demand
            ))

            # Debugging information
            print(f"Car: {car}, Current Market Trend: {current_trend}, Trend Factor: {trend_factor}, "
                  f"Demand: {exog_info['demand'][car]}, Safety Buffer: {safety_buffer}, Restock: {restock[car]}")

        # Set prices and discounts
        price = {car: max(0, self.model.state.competitor_price[car] - 50) for car in self.model.state.inventory_level}
        discount = {car: 0 for car in self.model.state.inventory_level}

        return Decision(restock=restock, price=price, discount=discount)

    def run_policy(self, policy_fn, n_iterations=10, **policy_args):
        """
        Executes a policy over multiple iterations.

        Args:
            policy_fn: Policy function.
            n_iterations: Number of iterations.
            policy_args: Additional arguments for the policy.

        Returns:
            List of results for each iteration.
        """
        results = []
        model_copy = deepcopy(self.model)

        for i in range(n_iterations):
            if model_copy.t >= model_copy.T:
                break

            # Generate exogenous information
            exog_info = model_copy.exog_info_fn(None)
            
            # Apply the policy function
            decision = policy_fn(theta_min=policy_args["theta_min"], theta_max=policy_args["theta_max"], exog_info=exog_info)

            # Transition state and calculate the new objective
            state_before = deepcopy(model_copy.state)
            new_state = model_copy.transition_fn(decision, exog_info)
            state_update = {key: new_state[key] for key in State._fields}
            model_copy.state = State(**state_update)
            step_profit = model_copy.objective_fn(decision, exog_info)
            model_copy.objective += step_profit

            # Profit components
            revenue = sum(min(model_copy.state.inventory_level[car], exog_info["demand"][car]) * decision.price[car]
                          for car in model_copy.state.inventory_level)
            restocking_cost = sum(decision.restock[car] * 300 for car in decision.restock)
            unsold_inventory = {
                car: model_copy.state.inventory_level[car] - min(model_copy.state.inventory_level[car], int(exog_info["demand"][car]))
                for car in model_copy.state.inventory_level
            }
            holding_cost = sum(unsold_inventory[car] * model_copy.state.holding_time[car] * self.model.holding_cost
                               for car in unsold_inventory)

            # Logging details
            print(f"Iteration {i}:")
            print(f"Market Trend: {state_before.market_trends}")
            print(f"Updated Market Trends: {new_state['market_trends']}")
            print("\nInventory Details:")
            print(f"Previous Inventory: {state_before.inventory_level}")
            print(f"Restock Decision: {decision.restock}")
            print(f"Demand: {exog_info['demand']}")
            print(f"Cars Sold: {new_state.get('cars_sold', {})}")
            print(f"Cars Added: {new_state.get('cars_added', {})}")
            print(f"New Inventory: {new_state['inventory_level']}")
            print("\nProfit and Costs:")
            print(f"Revenue: {revenue}, Restocking Cost: {restocking_cost}, Holding Cost: {holding_cost}, Profit: {step_profit}")
            print(f"Objective: {model_copy.objective}\n")

            # Append results
            results.append({
                "t": i,
                "state_before": state_before,
                "decision": decision,
                "state_after": new_state,
                "extra_info": {k: new_state[k] for k in new_state if k not in State._fields},
                "revenue": revenue,
                "restocking_cost": restocking_cost,
                "holding_cost": holding_cost,
                "profit": step_profit,
                "objective": model_copy.objective,
            })

        return results
