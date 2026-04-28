import torch 
import json
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import os


if __name__ == "__main__":  # this section will be not exported when we import our bert_model_evaluation class , the test_class has its own check for the gpu

 if torch.backends.mps.is_available():  # new instance of the same device declaration as in the training phase
    device = torch.device("mps")
    print("Uses Apple M1 8-core gpu for evaluation Dataset")
 else:
    device = torch.device("cpu")
    print("just uses cpu for evaluation")
    
    
class BertEvaluation:
    def __init__(self, model_path , device):
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained((model_path))
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, dataloader):
        predictions, true_labels, all_row_ids = [],[],[]
        
        print("Start of the evaluation ")
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                input_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                row_ids = batch[3]  # id origins
                
                outputs = self.model(input_ids, token_type_ids = None, attention_mask = input_mask)
                
                #"transformation" to send the gpu data back to the cpu (called logits), we minimize the data volume an get the raw data back. 
                logits = outputs.logits.detach().cpu().numpy()
                labels_ids = labels.to("cpu").numpy()
                
                batch_predictions = np.argmax(logits, axis = 1).flatten() # Choosing the one the 7 categories with the highest percentage of being right according to the predictions made by bert 
                
                predictions.extend(batch_predictions)          
                true_labels.extend(labels_ids)
                all_row_ids.extend(row_ids.cpu().numpy().tolist())  # if we would not apply this, we would get tensor data in the id column , for further steps we need the normal integer64 Datatype
                
            accuracy = accuracy_score(true_labels, predictions)
            
            
            df_validation = pd.DataFrame({
                'id': all_row_ids,
                'true_label' : true_labels,
                'prediction_bert':predictions
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
            
            df_validation['true_label'] = df_validation['true_label'].map(sentiment_prediction_maping_numeric_to_string)
            df_validation['prediction_bert'] = df_validation['prediction_bert'].map(sentiment_prediction_maping_numeric_to_string)
            
            report = classification_report(df_validation['true_label'],df_validation['prediction_bert'])
        
            
            return accuracy, report, df_validation
    
    def save_model_performance(self,save_path, accuracy,report,df_validation):
     
     results_summary ={
         "overall_accuracy" : accuracy,
         "classification_report" : report
     }
     
     with open(os.path.join(save_path, "evaluation_metrics_model.json"),"w") as evaluate_json:
         json.dump(results_summary, evaluate_json , indent=4)
     
     df_validation.to_csv(os.path.join(save_path, "model_performance.csv"), index=False)
    

if __name__ == "__main__":

 MODEL_PATH = "v6_lr2e-5_150warmupsteps_decay0.01_epochs5"       # ! Folder name check 

 full_model_path = f"./saved_models/{MODEL_PATH}"      

 model_pretrained = BertEvaluation(full_model_path,device)
 
 df_val_inputs = pd.read_parquet("./final_datasets/splitted_Datasets/validation_Datasets/ValidateQuestions.parquet")
 df_val_labels = pd.read_parquet("./final_datasets/splitted_Datasets/validation_Datasets/ValidateAnswers.parquet")

 input_ids = torch.tensor(df_val_inputs['tokenized_text'].to_list())
 attention_mask = (input_ids != 0).long()
 labels = torch.tensor(df_val_labels['sentiment'].to_list())
 row_ids = torch.tensor(df_val_inputs['id'].values).long()
                
 val_dataset = TensorDataset(input_ids, attention_mask, labels, row_ids)
 val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
 accuracy , report, df_validation = model_pretrained.evaluate(val_loader)
 
 MODEL_SAVE_PATH = f"./saved_models/{MODEL_PATH}"
 
 model_pretrained.save_model_performance( MODEL_SAVE_PATH, accuracy, report, df_validation)
 
# Informations directly in the terminal -> "Quick Check" 
 print(f"Acccuracy val dataset :{accuracy}")
 print(report)
 