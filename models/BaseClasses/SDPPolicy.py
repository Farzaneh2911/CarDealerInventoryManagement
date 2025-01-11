from copy import copy
from abc import ABC, abstractmethod
import pandas as pd
from . import SDPModel


class SDPPolicy(ABC):
    def __init__(self, model: SDPModel, policy_name: str = ""):
        self.model = model
        self.policy_name = policy_name
        self.results = pd.DataFrame()
        self.performance = pd.NA

    @abstractmethod
    def get_decision(self, state, t, T):
    
        
        
    
    # Seasonal thresholds
      current_month = t % 12
      is_winter = current_month in [9, 10, 11, 0, 1]
      theta_min = {"Car A": 5, "Car B": 10, "Car C": 5} if is_winter else {"Car A": 10, "Car B": 12, "Car C": 8}
      theta_max = {"Car A": 15, "Car B": 20, "Car C": 15} if is_winter else {"Car A": 20, "Car B": 25, "Car C": 20}

    # Adjust thresholds based on market trends
      if state.market_trends == "rising":
        theta_max = {car: value + 2 for car, value in theta_max.items()}
      elif state.market_trends == "declining":
        theta_min = {car: max(0, value - 2) for car, value in theta_min.items()}

    # Compute restocking levels
      restock = {}
      total_inventory = sum(state.inventory_level.values())
      available_capacity = self.model.max_inventory - total_inventory

      for car in state.inventory_level:
        if state.inventory_level[car] < theta_min[car]:
            # Target restock amount to bring inventory up to theta_max
            target_restock = theta_max[car] - state.inventory_level[car]
            # Ensure it does not exceed available capacity
            restock[car] = min(target_restock, available_capacity)
            available_capacity -= restock[car]
        else:
            restock[car] = 0

    # Pricing strategy: Adjust based on competitor prices and holding times
      price = {
        car: state.competitor_price[car] - (100 if state.holding_time[car] > 10 else 50)
        for car in state.inventory_level
      }

    # Discount strategy: Apply discounts for cars held over 30 days
      discount = {
        car: self.model.discount_rate if state.holding_time[car] > 30 else 0
        for car in state.inventory_level
      }

      return {"restock": restock, "price": price, "discount": discount}


    def run_policy(self, n_iterations: int = 1):
        """
        Runs the policy over the time horizon [0,T] for a specified number of iterations and return the mean performance.

        Args:
            n_iterations (int): The number of iterations to run the policy. Default is 1.

        Returns:
            None
        """
        result_list = []
        # Note: the random number generator is not reset when calling copy().
        # When calling deepcopy(), it is reset (then all iterations are exactly the same).
        for i in range(n_iterations):
            model_copy = copy(self.model)
            model_copy.episode_counter = i
            model_copy.reset(reset_prng=False)
            state_t_plus_1 = None
            while model_copy.is_finished() is False:
                state_t = model_copy.state
                decision_t = model_copy.build_decision(self.get_decision(state_t, model_copy.t, model_copy.T))

                # Logging
                results_dict = {"N": i, "t": model_copy.t, "C_t sum": model_copy.objective}
                results_dict.update(state_t._asdict())
                results_dict.update(decision_t._asdict())
                result_list.append(results_dict)

                state_t_plus_1 = model_copy.step(decision_t)

            results_dict = {"N": i, "t": model_copy.t, "C_t sum": model_copy.objective}
            if state_t_plus_1 is not None:
                results_dict.update(state_t_plus_1._asdict())
            result_list.append(results_dict)

        # Logging
        self.results = pd.DataFrame.from_dict(result_list)
        # t_end per iteration
        self.results["t_end"] = self.results.groupby("N")["t"].transform("max")

        # performance of one iteration is the cumulative objective at t_end
        self.performance = self.results.loc[self.results["t"] == self.results["t_end"], ["N", "C_t sum"]]
        self.performance = self.performance.set_index("N")

        # For reporting, convert cumulative objective to contribution per time
        self.results["C_t"] = self.results.groupby("N")["C_t sum"].diff().shift(-1)

        if self.results["C_t sum"].isna().sum() > 0:
            print(f"Warning! For {self.results['C_t sum'].isna().sum()} iterations the performance was NaN.")

        return self.performance.mean().iloc[0]