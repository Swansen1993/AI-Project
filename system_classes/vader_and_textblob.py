from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

vader_analyzer = SentimentIntensityAnalyzer()

df_training_vader_and_text_blob = pd.read_csv("./final_datasets/cleaned_data/cleaned_dataframe.csv")


df_training_vader_and_text_blob['vader_score'] = df_training_vader_and_text_blob['review_body'].apply(lambda x : vader_analyzer.polarity_scores(str(x))['compound'])
df_training_vader_and_text_blob['textblobscore']= df_training_vader_and_text_blob['review_body'].apply(lambda y : TextBlob(y).sentiment.polarity)
df_training_vader_and_text_blob[['vader_score', 'textblobscore']] = df_training_vader_and_text_blob[['vader_score', 'textblobscore']].round(3)

df_training_vader_and_text_blob.to_csv("./final_datasets/dataset_with_vader_and_textblob_score/dataset_with_vader_and_textblob_score.csv", index = False)