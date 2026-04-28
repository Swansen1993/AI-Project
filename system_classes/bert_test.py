import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from bert_model_evaluation import BertEvaluation 

if torch.backends.mps.is_available():  # new instance 
    device = torch.device("mps")
    print("Uses Apple M1 8-core gpu for test Dataset")
else:
    device = torch.device("cpu")
    print("just uses cpu for evaluation")


MODEL_PATH = "v6_lr2e-5_150warmupsteps_decay0.01_epochs5"         # ! Folder name check
       
evaluator_test_phase = BertEvaluation(f"saved_models/{MODEL_PATH}", device)    

df_test_input = pd.read_parquet("./final_datasets/splitted_Datasets/test_Dataset/TestQuestion.parquet")
df_test_labels = pd.read_parquet("final_datasets/splitted_Datasets/test_Dataset/TestAnswers.parquet")

input_ids = torch.tensor(df_test_input['tokenized_text'].to_list())
attention_mask = (input_ids != 0).long()
labels = torch.tensor(df_test_labels['sentiment'].to_list())
row_ids = torch.tensor(df_test_input['id'].values).long()

test_dataset =  TensorDataset(input_ids, attention_mask, labels, row_ids)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
accuracy, report, df_test_result = evaluator_test_phase.evaluate(test_loader)
    
    
print(f"Accuracy test dataset :{accuracy}")
print(report)

df_test_result.info()

df_test_result.to_csv("./final_datasets/splitted_Datasets/test_Dataset/test_results.csv",index=False) 
