import pandas as pd
import kagglehub
import os

# Direct Download of Datsets from Kaggle with the Kaggle Api and the Library kagglehub and convertion to a csv file
# the if not os.path exists was added because the kagglehub.datset_download method did not allow to have a existing file with that name in the output directory thats why i added a condition, this way we dont need to download the file every time 
if not os.path.exists('./datasets/amazon_reviews_us_Software_v1_00.tsv'):

 download = kagglehub.dataset_download(
     'cynthiarempel/amazon-us-customer-reviews-dataset', path ='amazon_reviews_us_Software_v1_00.tsv', output_dir= './datasets/raw_data'
    )
 
 tsv_file = './datasets/raw_data/amazon_reviews_us_Software_v1_00.tsv'
 csv_table = pd.read_table(tsv_file,sep='\t', on_bad_lines='warn')
 csv_table.to_csv('./datasets/raw_data/amazon_reviews_us_Software_v1_00.csv', index= False)


df_sentiment = pd.read_csv('./datasets/raw_data/amazon_reviews_us_Software_v1_00.csv')
df_sentiment = df_sentiment.loc[df_sentiment['review_body'].str.len()>25]
df_sentiment = df_sentiment.loc[df_sentiment['review_body'].str.len()<250]

df_sentiment = df_sentiment.drop(columns = ['marketplace','customer_id','review_id','product_id','product_parent','product_title','product_category','helpful_votes','total_votes','vine','verified_purchase','review_headline','review_date'])

# For This Dataset was no Sentiment avaiable, i created the column to train the model later more easily. 
df_sentiment.loc[df_sentiment['star_rating'].isin([4,5]) , 'sentiment'] = 'positive'
df_sentiment.loc[df_sentiment['star_rating'].isin ([3]) , 'sentiment'] = 'neutral'
df_sentiment.loc[df_sentiment['star_rating'].isin([1,2]) , 'sentiment'] = 'negative'


df_sentiment.to_csv("./datasets/processed_data/amazon_reviews_us_software_v1_preprocessed.csv",index=False)

if not os.path.exists('./datasets/raw_data/train-balanced-sarcasm.csv'):
    download = kagglehub.dataset_download(
        'danofer/sarcasm' , path= 'train-balanced-sarcasm.csv', output_dir= './datasets/raw_data'
    )
    
    
df_sarcasm = pd.read_csv('./datasets/raw_data/train-balanced-sarcasm.csv')

df_nosarcasm = df_sarcasm.loc[df_sarcasm['label'] == 0]
df_sarcasm = df_sarcasm.drop(df_nosarcasm.index)
df_sarcasm = df_sarcasm.loc[df_sarcasm['comment'].str.len()>100]
df_sarcasm = df_sarcasm.loc[df_sarcasm['comment'].str.len()<250]
df_sarcasm['sentiment'] = "sarcastic"
df_sarcasm = df_sarcasm.drop(columns= ['author','subreddit','score','ups','downs','date','created_utc','parent_comment','label'])
df_sarcasm.to_csv('./datasets/processed_data/justsarcasm.csv', index = True)


if not os.path.exists('./datasets/raw_data/train-irony.csv'):
    download = kagglehub.dataset_download(
        'nikhiljohnk/tweets-with-sarcasm-and-irony' , path= 'train.csv', output_dir= './datasets/raw_data'
    )

df_irony = pd.read_csv('./datasets/raw_data/train-irony.csv')
    
df_noIrony = df_irony.loc[df_irony['class'].isin(["figurative", "regular","sarcasm" ])]
df_irony = df_irony.drop(df_noIrony.index)
df_irony = df_irony.loc[df_irony['tweets'].str.len()>100]
df_irony = df_irony.loc[df_irony['tweets'].str.len()<200]
df_irony.to_csv('./datasets/processed_data/irony_preprocessed.csv', index= False)
