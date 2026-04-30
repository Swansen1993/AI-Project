import torch
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification
import tkinter as tk
from tkinter import filedialog
from torch.utils.data import TensorDataset, DataLoader
import datetime
from datetime import datetime
import matplotlib.pyplot as plt

if torch.backends.mps.is_available(): 
    device = torch.device("mps")
    print("Uses Apple M1 8-core gpu for evaluation Dataset")
else:
    device = torch.device("cpu")
    print("just uses cpu for evaluation")

class MarketingEvaluation:
    def __init__(self,model_path, device ):
     self.device = device
     self.model = BertForSequenceClassification.from_pretrained((model_path))
     self.model.to(self.device)
     self.model.eval()
     self.input_path = None

     
    def file_selector(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
        title = "Choose the file with the customer review data, just parquet files are allowed",
        filetypes=[("Allowed data format", "*.parquet")]
        )
        
        root.destroy()
        
        self.input_path = file_path
        
        if file_path.endswith('parquet'):
            self.df_marketing_predictions = pd.read_parquet(file_path)
            
        print(f"File was loaded{len(self.df_marketing_predictions)} reviews found")
        
        return self.df_marketing_predictions
     
    
    
    def evaluate_marketing_data(self, dataloader):
        predictions_bert, all_row_ids = [],[]
        
        print("Start of the Customer Review analysis")
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                input_mask = batch[1].to(self.device)
                row_ids = batch[2] 
                
                outputs = self.model(input_ids, token_type_ids = None, attention_mask = input_mask)
                
                logits = outputs.logits.detach().cpu().numpy()
                batch_predictions = np.argmax(logits, axis = 1).flatten()  
                
                predictions_bert.extend(batch_predictions)          
                all_row_ids.extend(row_ids.cpu().numpy().tolist())  
                
            df_marketing_predictions_bert = pd.DataFrame({
                'id': all_row_ids,
                'prediction_bert':predictions_bert
            })
            
            sentiment_prediction_maping_numeric_to_string = {
            0: "negative",
            1: "neutral",
            2: "positive", 
            3: "negation",
            4: "multipolarity",
            5: "sarcastic", 
            6: "irony"
            }
            
            df_marketing_predictions_bert['prediction_bert'] = df_marketing_predictions_bert['prediction_bert'].map(sentiment_prediction_maping_numeric_to_string)
            
            df_marketing_with_review_text = pd.merge(
                df_marketing_predictions_bert,
                self.df_marketing_predictions [['id', 'review_body']],
                on='id',
                how='left'
            )
                
            return  df_marketing_with_review_text
        
    def save_customer_review_predciton(self,df_marketing_with_review_text):
        
        timestamp = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
        
        archive_path = f"./final_datasets/tableu_data/archive_data/tableu_export_{timestamp}.csv"
        df_marketing_with_review_text.to_csv(archive_path, index=False)
        
        live_path = "./final_datasets/tableu_data/live_data/marketing_current_data.csv"
        df_marketing_with_review_text.to_csv(live_path, index=False)
        
        
MODEL_FIT = "v3_lr2e-5_100warmupsteps_0.01weight_decay"
MODEL_USED = f"./saved_models/{MODEL_FIT}"

marketing_input_and_model = MarketingEvaluation(MODEL_USED,device)

# using our file selector
file_input = marketing_input_and_model.file_selector()

#constraint is that the file needs to be allready in a certain format when the employee of the marketing department feeds it to our system. 
input_ids = torch.tensor(file_input['tokenized_text'].to_list())
attention_mask = (input_ids != 0).long()
row_ids = torch.tensor(file_input['id'].values).long()

customer_reviews = TensorDataset(input_ids, attention_mask, row_ids)
customer_reviews_dataloader = DataLoader(customer_reviews, batch_size=16, shuffle= False)

df_customer_review_for_tableu = marketing_input_and_model.evaluate_marketing_data(customer_reviews_dataloader)

marketing_input_and_model.save_customer_review_predciton(df_customer_review_for_tableu)
