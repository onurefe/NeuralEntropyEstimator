from matplotlib import pyplot as plt
import pandas as pd

df1 = pd.read_csv("transient_entropies_hidden2_4dims.csv")
df2 = pd.read_csv("transient_entropies_hidden4_4dims.csv")
df3 = pd.read_csv("transient_entropies_hidden8_4dims.csv")
df = pd.DataFrame({"Hidden2":df1["Estimated entropies"], 
                   "Hidden4":df2["Estimated entropies"],
                   "Hidden8":df3["Estimated entropies"]})

# Plotting
plt.figure(figsize=(12, 6))

for column in df.columns:
    plt.plot(df.index, df[column], label=column)

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of DataFrame Columns')

# Adding legend
plt.legend()

# Show plot
plt.savefig("neural_pdf_multiple.png")