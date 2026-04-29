import pandas as pd

# i added the exported csv file from label studio unter "./final_datasets/sample_data_label_studio/dataset_after_manual_anotation_label_studio.csv"
df_manual_labeled = pd.read_csv("./final_datasets/sample_data_label_studio/dataset_after_manual_anotation_label_studio.csv")
df_real_data = pd.read_csv("./final_datasets/sample_data_label_studio/sample_data_check_label_studio.csv")


df_manual_labeled = df_manual_labeled.drop(columns= {"annotation_id","annotator","created_at","id","lead_time","review_body","updated_at"})
df_manual_labeled = df_manual_labeled.rename(columns= {"sentiment": "manual_label"})
df_manual_labeled.to_csv("./final_datasets/sample_data_label_studio/labeled_data_after_removed_columns_of_original_data.csv", index= False)

df_comparison = pd.merge(df_manual_labeled[['ID', 'manual_label']], 
                         df_real_data[['ID', 'sentiment']], 
                         on='ID')

df_comparison['manual_label'] = df_comparison['manual_label'].str.lower().str.strip()
df_comparison['sentiment'] = df_comparison['sentiment'].str.lower().str.strip()


df_comparison['matching'] = df_comparison['manual_label'] == df_comparison['sentiment']
#creating csv file for manual check
df_comparison.to_csv("./final_datasets/sample_data_label_studio/matching.csv", index = False)

accuracy = df_comparison['matching'].mean() * 100 
print(f"Human Labeled Data Accuracy : {accuracy: .2f}%")

# The Accuracy of 70,41% shows the complexity of the Dataset, while labeling i had Problem with distinguishing the Categories, especially with the categories irony and sarcasm. Also the Category Negation and Neutral are a challenge
# This reflects the real world becuase sarcasm and irony are adjacent lingustical concepts, same goes for negative and negational patterns. 
# With these Results we are Setting a Human Baseline the System should at least have a accuracy of 65% 