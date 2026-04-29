import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path

# Filtering of the Positive and Negative Reviews, we want to get 2000 of each Category. These two Categories will have the most Data because they are easy to obtain and very Common in the real World. 
df = pd.read_csv("./datasets/amazon_reviews_us_software_v1_preprocessed.csv") 
dfshuffled = df.sample(frac=1) # Randomly Shuffle the data to achieve a higher data Quality. Reviews are now more random and broad.  
 
df_positiveReviews = dfshuffled
#df_positiveReviews.info() # Checking which Datatype every Column has, this can affect how we need to apply our filters 

positivesummaryPatterns = "good|nice|awesome|perfect|excellent|great"

df_positiveReviews = df_positiveReviews.loc[(df_positiveReviews['star_rating'].isin([4,5])) & (df_positiveReviews['review_body'].str.contains(positivesummaryPatterns,na= False, case= False))].head(1500) 
df_positiveReviews = df_positiveReviews.drop(columns ='star_rating') # We dint need the Rating anymore and our other Dataframes csv irony and sarcasm dont have this column for the joining of these csv to a Dataframe we need to remove the star_rating
df_positiveReviews.to_csv("final_datasets/all_categories_after_filtering/PositiveReviewsAll.csv", index_label="ID") # Naming the index Column to identify it in the next steps 

dfshuffled = dfshuffled.drop(df_positiveReviews.index) # The Index Numbers of the Created Dataset are getting removed from the Initial DataSet we can now Extract a Other Category 
dfshuffled.to_csv("datasets/big_dataset_after_removals/DatasetPosRm.csv", index_label="ID") # Saving the Initial Dataset after Removing the Positive Reviews 


df_negation = pd.read_csv("datasets/DatasetPosRm.csv", index_col=0) # setting index_col=0 so that pandas does not add a new id column
df_negationReviews = df_negation.copy()

negationsummaryPatterns = "No|Never|Not|wasn't"

df_negationReviews = df_negationReviews.loc[(df_negationReviews['star_rating'].isin([1,2,3])) & (df_negationReviews['review_body'].str.contains(negationsummaryPatterns,na= False, case=False))].head(1000) 
df_negationReviews["sentiment"] = df_negationReviews["sentiment"].replace("negative","negation")
df_negationReviews["sentiment"] = df_negationReviews["sentiment"].replace("neutral","negation")
df_negationReviews = df_negationReviews.drop(columns ='star_rating') 
df_negationReviews.to_csv("final_datasets/all_categories_after_filtering/NegationReviewsAll.csv")

df_negation = df_negation.drop(df_negationReviews.index)
df_negation.to_csv("datasets/big_dataset_after_removals/DatasetPosNegRm.csv")


df_negative = pd.read_csv("datasets/DatasetPosNegRm.csv", index_col=0)
df_negativeReviews = df_negative.copy()

negativesummaryPatterns = "horrible|bad|hate|waste|broken|rubbish|don't|doesn't"

df_negativeReviews = df_negativeReviews.loc[(df_negativeReviews['star_rating'].isin([1,2])) & (df_negativeReviews['review_body'].str.contains(negativesummaryPatterns,na= False, case=False))].head(1500) 
df_negativeReviews = df_negativeReviews.drop(columns='star_rating')
df_negativeReviews.to_csv("final_datasets/all_categories_after_filtering/NegativeReviewsAll.csv")

df_negative = df_negative.drop(df_negativeReviews.index)
df_negative.to_csv("datasets/big_dataset_after_removals/DatasetPosNegNegativeRm.csv")

df_neutral = pd.read_csv("datasets/DatasetPosNegNegativeRm.csv", index_col=0)
df_neutralReviews = df_neutral.copy()

neutralsummaryPatterns = "ok|fair|mediocre"
forbiddenpattern= "but| But"

df_neutralReviews = df_neutralReviews.loc[(df_neutralReviews['star_rating'].isin([3])) & (df_neutralReviews['review_body'].str.contains(neutralsummaryPatterns,na= False, case=False)) & ~ df_neutralReviews["review_body"].str.contains(forbiddenpattern)].head(1000)
df_neutralReviews = df_neutralReviews.drop(columns='star_rating') 
df_neutralReviews.to_csv("final_datasets/all_categories_after_filtering/NeutralReviewsAll.csv")


df_neutral = df_neutral.drop(df_neutralReviews.index)
df_neutral.to_csv("datasets/big_dataset_after_removals/DatasetPosNegNegativeNeutRm.csv")


df_multi = pd.read_csv("datasets/DatasetPosNegNegativeNeutRm.csv", index_col=0)
df_multipolarityReviews = df_multi.copy()

multipolritysummaryPatterns = "but | But"

df_multipolarityReviews = df_multipolarityReviews.loc[(df_multipolarityReviews['star_rating'].isin([3])) & (df_multipolarityReviews['review_body'].str.contains(multipolritysummaryPatterns,na= False, case=False))].head(1000) 
df_multipolarityReviews["sentiment"] = df_multipolarityReviews["sentiment"].replace("neutral", "multipolarity") 
df_multipolarityReviews = df_multipolarityReviews.drop(columns='star_rating')
df_multipolarityReviews.to_csv("final_datasets/all_categories_after_filtering/MultipolarReviewsAll.csv")


df_multi = df_multi.drop(df_multipolarityReviews.index)
df_multi.to_csv("datasets/big_dataset_after_removals/DatasetPosNegNegativeNeutMultiRm.csv")

df_sarcasm = pd.read_csv("datasets/justsarcasm.csv", index_col=0)
df_sarcasmReviews = df_sarcasm.copy()

sarcasmPatterns = "Yeah,okay|yeah,okay|GREAT|NICE|THANKS|GOOD"

df_sarcasmReddit = df_sarcasmReviews.loc[df_sarcasmReviews['comment'].str.contains(sarcasmPatterns, na=False, case =False )].head(3000)
df_sarcasmReddit = df_sarcasmReddit.rename(columns={"comment" : "review_body"})
df_sarcasmReddit.to_csv("final_datasets/all_categories_after_filtering/SarcasticReviewsAll.csv", index_label="ID")


df_irony = pd.read_csv("datasets/irony_preprocessed.csv")
df_ironicReviews = df_irony.copy()
df_ironicReviews.info()


ironicPattern = "Yeah,okay|yeah,okay|GREAT|NICE|THANKS|GOOD|AMAZING|FANTASTIC|WONDERFUL|BRILLIANT|OUTSTANDING|REALLY|OH REALLY|SHOCKER|SURPRISE SURPRISE|LOVE IT"

df_ironicReviews = df_ironicReviews.loc[df_ironicReviews['tweets'].str.contains(ironicPattern, na=False, case =False)]
df_ironicReviews['tweets'] = df_ironicReviews['tweets'].astype(str).str.replace('#\w+', '', regex=True)
df_ironicReviews['tweets'] = df_ironicReviews['tweets'].replace('http\S+|www\S+',' ', regex=True)
df_ironicReviews["tweets"] = df_ironicReviews["tweets"].astype(str).str.replace("\\n"," ", regex=True)
df_ironicReviews['tweets'] = df_ironicReviews['tweets'].replace('@\w+',' ', regex= True).str.strip()
df_ironicReviews= df_ironicReviews.rename(columns={"tweets":"review_body","class":"sentiment"})
df_ironicReviews.to_csv("final_datasets/all_categories_after_filtering/IronicReviewsAll.csv", index_label="ID")

# Alltogether we can just extract 878 Reviews instead of a 1000. This a small Constraint that we can accept. 

path = './final_datasets/all_categories_after_filtering'
all_files = glob.glob(os.path.join(path, "*csv"))

data_frames =[]
total_rows_files = 0 

print("Length of the Data_files")

for f in all_files: 
    temp_df = pd.read_csv(f)
    
    current_rows = temp_df.shape[0]
    total_rows_files += current_rows
    
    print("File:{0} | Rows:{1}".format(os.path.basename(f),current_rows))
    
    data_frames.append(temp_df)
    
df_all_data = pd.concat(data_frames, ignore_index = True)


df_all_data.to_csv("./final_datasets/all_reviews/all_reviews.csv", index = False)
