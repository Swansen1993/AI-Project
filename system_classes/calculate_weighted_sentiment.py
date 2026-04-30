from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report


def calculate_weighted_sentiment(vader_score, textBlob_score, bert_class ):
    complex_mapping = {
    3:"negation" ,
    4:"multipolarity",
    5:"sarcasm" ,
    6:"irony" 
    }
    
    if bert_class in complex_mapping:
        return complex_mapping[bert_class]
    
    if bert_class == 3:
        return "negation"
    elif bert_class == 4:
        return "multipolarity"
    elif bert_class == 5:
        return "sarcasm"
    elif bert_class == 6:
        return "irony"
        
    bert_score =(bert_class-3)/3.0     
    
    weight_textblob = 1.0
    weight_vader = 1.5
    weight_bert = 2.5
    
    total_weight = weight_textblob + weight_vader + weight_bert
        
    weighted_score =((textBlob_score * weight_textblob) +
                     (vader_score * weight_vader)+
                     (bert_score * weight_bert))  / total_weight
     
     
    if weighted_score >=0.05:
        return "positive", weighted_score
    if weighted_score <= -0.05:
        return "negative", weighted_score
    else:
        return "neutral", weighted_score
    

    
df_textblob_vader= pd.read_csv("./final_datasets/dataset_weighted_scores/weighted_scores.csv")

df_bert = pd.read_csv("final_datasets/splitted_Datasets/test_Dataset/test_results.csv")

if 'prediction' in df_textblob_vader.columns:
    df_textblob_vader = df_textblob_vader.drop(columns=['prediction']) # because of the merge method this is integrated, wo stop merge from creating new prediction columns
    

df_final = df_textblob_vader.merge(df_bert[['id', 'prediction']], on='id', how='inner')
df_final = df_final.drop_duplicates(subset=['id'], keep='first') # some reviews had duplicates after merge, removing them 


if 'hybrid_label' in df_final.columns:
    df_final = df_final.drop(columns=['hybrid_label', 'hybrid_score']) 

results = df_final.apply(
    lambda x: calculate_weighted_sentiment(
        vader_score =x['vader_score'],
        textBlob_score=x['textblobscore'],
        bert_class = x['prediction']
        ), 
    axis=1
)

df_final['hybrid_label'] = results.apply(lambda x: x[0] if isinstance(x, (tuple, list)) else x)
df_final['hybrid_score'] = results.apply(lambda x: x[1] if isinstance(x, (tuple, list)) else 0.0)

df_final[['weighted_score']] = df_final[['weighted_score']].round(3)
df_final[['hybrid_score']] = df_final[['hybrid_score']].round(3)


# we need this map to map the predictions from the bert model in the column prediciton because they are still in a numeric form
sentiment_prediction_maping_numeric_to_string = {
    0: "negative",
    1: "neutral",
    2: "positive", 
    3: "negation",
    4: "multipolarity",
    5: "sarcastic", 
    6: "irony"
}


df_prediction_numeric_to_string = df_final.copy()
df_prediction_numeric_to_string['prediction'] =  df_prediction_numeric_to_string['prediction'].map(sentiment_prediction_maping_numeric_to_string)

accuracy_vader_textBlob_all_Categories = accuracy_score( df_final['sentiment'],df_final['weighted_sentiment_label'])
accuracy_all_systems_allCategories = accuracy_score(df_final['sentiment'], df_final['hybrid_label'])
accuracy_just_Bert_all_Categories = accuracy_score(df_prediction_numeric_to_string['sentiment'], df_prediction_numeric_to_string['prediction']) 


print(f"Accuracy VADER + TextBlob all categories: {accuracy_vader_textBlob_all_Categories:.4f}")
print(f"Accuracy Vader + TextBlob +Bert all categories : {accuracy_all_systems_allCategories:.4f}")
print(f"Accuracy just Bert all categories: {accuracy_just_Bert_all_Categories:.4f}")


#Spltting the Data in base Categories and Complex Categories for further insights
target_labels = ["positive", "neutral", "negative"]
df_filtered_base_categories = df_final[df_final['sentiment'].isin(target_labels)].copy()

target_labels_complex = ["negation", "multipolarity", "sarcastic", "irony"]
df_filtered_complex_categories = df_final[df_final['sentiment'].isin(target_labels_complex)].copy()

#maping both Dataframes base and complex, this way we can use the prediction column to track the performance of the bert model and to compare performances between bert and vader/TextBlob
df_prediction_mapped_base_category = df_filtered_base_categories.copy()
df_prediction_mapped_base_category['prediction'] = df_prediction_mapped_base_category['prediction'].map(sentiment_prediction_maping_numeric_to_string)

df_predcition_mapped_complex_categories = df_filtered_complex_categories.copy()
df_predcition_mapped_complex_categories['prediction'] = df_predcition_mapped_complex_categories['prediction'].map(sentiment_prediction_maping_numeric_to_string)

accuracy_base_categories_vader_textBlob = accuracy_score(df_filtered_base_categories['sentiment'],df_filtered_base_categories['weighted_sentiment_label'])
accuracy_base_categories_allSystems = accuracy_score(df_filtered_base_categories['sentiment'],df_filtered_base_categories['hybrid_label'])
accuracy_base_categories_justBert = accuracy_score(df_prediction_mapped_base_category['sentiment'],df_prediction_mapped_base_category['prediction'])
accuracy_complex_categories_justBert = accuracy_score(df_predcition_mapped_complex_categories['sentiment'], df_predcition_mapped_complex_categories['prediction'])


print(f"Accuracy VADER + TextBlob base categories(positive, neutral, negative): {accuracy_base_categories_vader_textBlob:.4f}")
print(f"Accuracy Vader + TextBlob +  BERT base categories(positive, neutral, negative): {accuracy_base_categories_allSystems:.4f}")
print(f"Accuracy base categories just Bert (positive, neutral, negative): {accuracy_base_categories_justBert:.4f}")
print(f"Accuracy complex categories Bert (negation,multipolarity,sarcastic,irony): {accuracy_complex_categories_justBert:.4f}")

df_final.to_csv("./final_datasets/dataset_weighted_scores/weighted_scores.csv", index= False)
df_filtered_base_categories.to_csv("./final_datasets/dataset_weighted_scores/base_categories.csv", index =False)
df_prediction_mapped_base_category.to_csv("./final_datasets/dataset_weighted_scores/bert_base_categories.csv", index =False)
df_predcition_mapped_complex_categories.to_csv("./final_datasets/dataset_weighted_scores/bert_complex_categories.csv", index = False)   