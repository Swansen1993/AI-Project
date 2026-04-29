Ressources and external Data: 

Caused by the File Size some files are not uploaded in this Github Repository. This concerns the trained Bert models and Raw Data Files from Kaggle. 

Trained Models can be found here: 
https://huggingface.co/Swansen1993/RepositoryBertModels/tree/main

The Used Raw Data Files from Kaggle can be found here:

dataset neutral,negative,positive,multipolarity, negation: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon_reviews_us_Software_v1_00.tsv
sarcastic reviews : https://www.kaggle.com/datasets/danofer/sarcasm
ironic reviews : https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony

The Code in the Class dataset_download_and_preprocessing should work to download these directly from Kaggle. 
As a safety precaution the links will be posted here anyways. 

In the Folder for Phase 2 is every Data File that is missing in the Github Repository these Files can also be added manually 

file path bert models: ./AI Project /saved_models/ Folder with the models 
file path raw data : ./AI Project /datasets/raw_data/ raw_data files

This way all paths in the classes should work. 

Short User Guide Tableu new Data: 

The System promises a renewal of data after a new file was processed with the marketing_datasets_input_User_Interface class. 

We have 1 challenge that hinders us to deliver the functionality directly with the shipped tableu package of phase 2. 
The deeper problem is that in tableu a normal package tableu.twb is indeed dynamic and can adjust the visualized data after we start the marketing_datasets_input_User_Interface class

But in the context of this university project i need to ship the tableu package to my tutor. The twb package is very fragile, the paths to load the csv-Files to load the new data is a path that is a full path, so it also includes the username. 

To still be able to use the dynamic functions of the tableu dashboard i recommend following approach/workaround: 

1. Open the Tableu Sentiment Customer Review Project.twbx. 
2. Save under "twb." -> save in the same Folder "Phase 2 all Files/Data ->  new file with Tableu Sentiment Customer Review Project.twb will be created  
3. Open the twb. file -> click data source -> click marketing_current data -> and navigate to the  AI Project Folder/final_datasets/tableu_data/live_data/markting_current_data.csv -> This will link the folder where we actually create the new files with our marketing_datasets_input_User_Interface class 
4. make sure "live" is choosen in the upper right. 
5. refresh the data source in the top left 

The Dataset loaded right now into the tableu Dashboard is the : /AI Project /final_datasets/splitted_Datasets/test_Dataset/TestQuestion.parquet
So for further testing you should try the dataset : /AI Project /final_datasets/splitted_Datasets/validation_Datasets/ValidateQuestions.parquet 

This approach is pretty complicated but the only one i could find where i can guarantee a secure shippment of the tableu package and also being able to show the dynamic functionality. 