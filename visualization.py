import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("data/bioprocess_dataset.csv")

plt.figure(figsize=(8,5))

plt.scatter(
    dataset["fermentation_time_h"],
    dataset["product_yield_g_l"]
)

plt.xlabel("Fermentation Time (h)")
plt.ylabel("Product Yield (g/L)")
plt.title("Effect of Fermentation Time on Product Yield")

plt.tight_layout()

plt.savefig(
    "figures/fermentation_time_vs_yield.png",
    dpi=300
)

plt.show()


plt.figure(figsize=(8,5))

plt.scatter(
    dataset["temperature_c"],
    dataset["product_yield_g_l"]
)

plt.xlabel("Temperature (°C)")
plt.ylabel("Product Yield (g/L)")
plt.title("Effect of Temperature on Product Yield")

plt.tight_layout()

plt.savefig(
    "figures/temperature_vs_yield.png",
    dpi=300
)

plt.show()