import matplotlib.pyplot as plt
import pandas as pd

df_plot = pd.read_csv("./final_datasets/tableu_data/live_data/marketing_current_data.csv")
sentiment_counts = df_plot['prediction_bert'].value_counts()

plt.figure(figsize=(16,9))
sentiment_counts.plot(kind = 'bar', color = ['green', 'blue','red', 'yellow', 'grey', 'orange', 'purple'])
plt.title("Quick-Check: Categorial Distribution")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.grid(axis ='y', linestyle='--', alpha=0.7)
plt.show()