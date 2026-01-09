import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curves(file_path):
    # Load the data
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("CSV file not found. Make sure you've run the training first!")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 1. Plot Total Reward (Learning Curve)
    ax1.plot(data['Episode'], data['Total_Reward'], color='blue', alpha=0.3)
    # Adding a rolling average to see the trend through the noise
    ax1.plot(data['Episode'], data['Total_Reward'].rolling(window=10).mean(), color='red', linewidth=2)
    ax1.set_title("Total Reward per Episode (Red = Trend)")
    ax1.set_ylabel("Reward")
    ax1.grid(True)

    # 2. Plot SoC (Efficiency)
    ax2.plot(data['Episode'], data['Avg_SoC'], color='green')
    ax2.axhline(y=0.2, color='orange', linestyle='--', label='Low Battery Threshold')
    ax2.set_title("Average State of Charge (SoC)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("SoC (0.0 to 1.0)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_learning_curves("ev_marl_results.csv")