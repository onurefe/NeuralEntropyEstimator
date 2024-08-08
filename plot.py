from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")


"""df1 = pd.read_csv("transient_entropies_diagonal_kernel.csv")
df2 = pd.read_csv("transient_entropies_kernel.csv")
df3 = pd.read_csv("transient_entropies_neural.csv")
df = pd.DataFrame({"Diagonal kernel":df1["Estimated entropies"], 
                   "Kernel":df2["Estimated entropies"],
                   "Neural":df3["Estimated entropies"],
                   "True entropy": df3["True entropies"]})
"""
# Plotting
plt.figure(figsize=(12, 6))

for column in df.columns[1:]:
    plt.plot(df.index, df[column], label=column)

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of DataFrame Columns')

# Adding legend
plt.legend()

# Show plot
plt.savefig("image.png")