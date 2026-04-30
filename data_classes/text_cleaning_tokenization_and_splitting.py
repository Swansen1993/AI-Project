import pandas as pd 
import janitor
import re
import demoji
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split



def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = demoji.replace(text, "")
    text = re.sub(r'[a-zA-Z\s]]', '', text)
    return text.lower().strip()



# final Cleaning of our Datframe, especially the review_body column needs to be perfectly prepared before we tokenize the data. We want to reduce the noise here. 

# Method chaining with pyjanitor
df_cleaned = (
    pd.read_csv("./final_datasets/all_reviews/all_reviews.csv")
    .clean_names()
    .remove_empty()
    .transform_column("review_body", clean_text, elementwise =True)
)

df_cleaned.to_csv("final_datasets/cleaned_data/cleaned_dataframe.csv", index = False)

review_lengths = df_cleaned['review_body'].str.split().str.len()
print(review_lengths.describe()) # figuring out how long our max length of the strings should be. This way we can avoid to much padding. 


#vertically splitting the Dataset to hold the answers and questions separate
# = df_cleaned['review_body], df_cleaned['sentiment']

# A 70/15/15 Split is the Goal, We first Split the Train and Rest Size 

X_all_data_questions = df_cleaned[["id","review_body"]]
y_all_data_answers = df_cleaned["sentiment"]


X_TrainQuestion, X_tempQuestions, y_TrainAnswers, y_tempAnswers = train_test_split (
    X_all_data_questions, y_all_data_answers, 
    test_size = 0.3, 
    random_state= 42,
    stratify= y_all_data_answers  
)

# ensuring that the answer_id for every is the same as the question_id
y_train_answers_with_ID = pd.DataFrame({
    "id" : X_TrainQuestion ["id"],
    "sentiment" : y_TrainAnswers
})

#Bert Function to tokenize the review body text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(text):
    return tokenizer.encode(text, add_special_tokens=True, max_length=64, truncation=True, padding='max_length')


#label dictonary for the answers 
label_dict ={'negative' :0, 'neutral' :1, 'positive' :2, 'negation':3, 'multipolarity':4, 'sarcastic':5, 'irony':6}


X_TrainQuestion['tokenized_text'] = X_TrainQuestion['review_body'].apply(tokenize_function)
y_train_answers_with_ID['sentiment'] = y_train_answers_with_ID['sentiment'].map(label_dict)

X_TrainQuestion.to_csv("final_datasets/splitted_Datasets/training_Dataset/checktrainingQuestionFormat.csv", index = False)
y_train_answers_with_ID.to_csv("final_datasets/splitted_Datasets/training_Dataset/checktrainingAnswerFormat.csv", index = False)

X_TrainQuestion.to_parquet("./final_datasets/splitted_Datasets/training_Dataset/TrainingQuestions.parquet", index=False)
y_train_answers_with_ID.to_parquet("./final_datasets/splitted_Datasets/training_Dataset/TrainingAnswers.parquet",index=False)


X_ValidateQuestion, X_TestQuestion, y_ValidateAnswer, y_TestAnswer = train_test_split(
   X_tempQuestions , y_tempAnswers,
   test_size = 0.5, # Splitting the Validation and Test Set 
   random_state=42,
   stratify= y_tempAnswers
)


y_validate_answers_with_ID = pd.DataFrame({
    "id" : X_ValidateQuestion["id"],
    "sentiment" : y_ValidateAnswer
})

X_ValidateQuestion['tokenized_text'] = X_ValidateQuestion['review_body'].apply(tokenize_function)
y_validate_answers_with_ID['sentiment'] = y_validate_answers_with_ID['sentiment'].map(label_dict) 

X_ValidateQuestion.to_csv("./final_datasets/splitted_Datasets/validation_Datasets/checkValidationQuestionFormat.csv", index = False)
y_validate_answers_with_ID.to_csv("./final_datasets/splitted_Datasets/validation_Datasets/checkValidationAnswersFormat.csv", index = False)


X_ValidateQuestion.to_parquet("./final_datasets/splitted_Datasets/validation_Datasets/ValidateQuestions.parquet", index=False)
y_validate_answers_with_ID.to_parquet("./final_datasets/splitted_Datasets/validation_Datasets/ValidateAnswers.parquet", index=False)


y_test_answers_with_id = pd.DataFrame({
    "id" : X_TestQuestion["id"],
    "sentiment" : y_TestAnswer
})

X_TestQuestion['tokenized_text'] = X_TestQuestion['review_body'].apply(tokenize_function)
y_test_answers_with_id['sentiment'] = y_test_answers_with_id['sentiment'].map(label_dict)

#csv Check 
X_TestQuestion.to_csv("./final_datasets/splitted_Datasets/test_Dataset/checktestquestionformat.csv", index = False)
y_test_answers_with_id.to_csv("./final_datasets/splitted_Datasets/test_Dataset/checktestanwers.csv", index = False)

#parquet Files for Training of the bert model
X_TestQuestion.to_parquet("final_datasets/splitted_Datasets/test_Dataset/TestQuestion.parquet",index=False)
y_test_answers_with_id.to_parquet("final_datasets/splitted_Datasets/test_Dataset/TestAnswers.parquet",index=False)

print((y_TrainAnswers).value_counts())  # Check if the Datsets are balanced and has every category in the right distribution weight. 
print((y_ValidateAnswer).value_counts())
print((y_TestAnswer).value_counts())
