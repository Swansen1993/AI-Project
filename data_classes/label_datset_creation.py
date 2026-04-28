import pandas as pd
from sklearn.model_selection import train_test_split


df_cleaned = pd.read_csv("final_datasets/cleaned_data/cleaned_dataframe.csv")

# Datframe is getting split vertically to separate the review Text from the sentiment, this way i can label the data without any bias. 
X_all_data_questions = df_cleaned[["id","review_body"]]
y_all_data_answers = df_cleaned["sentiment"]

y_all_data_answers_with_ID = pd.DataFrame({
    "id" : X_all_data_questions ["id"],
    "sentiment" : y_all_data_answers
})

X_LabelSample, _, y_LabelSample, _ = train_test_split(
    X_all_data_questions, 
    y_all_data_answers_with_ID, 
    train_size=500, 
    random_state=42, 
    stratify = y_all_data_answers
)

X_LabelSample.to_csv("./final_datasets/sample_data_label_studio/sample_data_label_studio2.csv", index=False)
y_LabelSample.to_csv("final_datasets/sample_data_label_studio/sample_data_check_label_studio2.csv",index =False)