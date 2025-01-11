import numpy as np
import pandas as pd
from CarDealerModel import CarDealerInventoryModel
from policy import CarDealerPolicy

def load_initial_state(file_path):
    # Read the file into a DataFrame
    df = pd.read_csv(file_path)  # Change to pd.read_excel for Excel files

    # Process the DataFrame into the required format
    S0 = {
        "inventory_level": dict(zip(df["Model"], df["Number in Stock"])),
        "holding_time": dict(zip(df["Model"], df["Holding_Time(Week)"])),
        "competitor_price": dict(
            zip(
                df["Model"], 
                df["Selling_Price"]
                .str.replace('£', '', regex=False)
                .str.replace(',', '', regex=False)
                .astype(float)
            )
        ),
        "market_trends": {model: "stable" for model in df["Model"]},  # Assuming all trends are stable initially
        "demand_forecast": {model: 10 for model in df["Model"]},  # Placeholder demand forecast
    }

    # Dynamically create thresholds 
    theta_min = {model: max(2, stock - 2) for model, stock in S0["inventory_level"].items()}
    theta_max = {model: stock + 5 for model, stock in S0["inventory_level"].items()}

    return S0, theta_min, theta_max

def main():
    # Path to the data file
    file_path = "/Users/farzanehhaghighatbin/Desktop/Car Dealer/data/initial_state.csv"  # Replace with your actual file path

    # Load the initial state
    S0, theta_min, theta_max = load_initial_state(file_path)

    T = 30  # Time horizon

    # Initialize the model and policy
    model = CarDealerInventoryModel(S0=S0, T=T, max_inventory=50, market_trends_file="/Users/farzanehhaghighatbin/Desktop/Car Dealer/data/cleaned_market_trends.csv")
    policy = CarDealerPolicy(model=model, policy_name="OrderUpToPolicy")

    # Test the policy
    print("\n--- Testing Basic Order-Up-To Policy ---")
    results = policy.run_policy(
        policy_fn=policy.basic_order_up_to_policy,
        n_iterations=10,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    # Output results
    for result in results:
        print(f"Time {result['t']}: Objective: {result['objective']}, Restock: {result['decision'].restock}")

if __name__ == "__main__":
    main()