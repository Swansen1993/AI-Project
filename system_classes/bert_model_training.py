import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.optim as optimizer
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
import os


model = BertForSequenceClassification.from_pretrained(      #defining the bert model we want to use 
    'bert-base-uncased',
    num_labels = 7,
    output_hidden_states = False, # Saving gpu memory, we dont need the internal details 
    output_attentions = False
)

if torch.backends.mps.is_available():  # for the training of our model we use the Gpu of my macbook, as a model training should always be done with the gpu instead of the cpu. 
    device = torch.device("mps")
    print("Uses Apple M1 8-core gpu")
else:
    device = torch.device("cpu")
    print("just uses cpu")

model.to(device) # exporting the model to the gpu 

df_train_inputs = pd.read_parquet("./final_datasets/splitted_Datasets/training_Dataset/TrainingQuestions.parquet")
df_train_labels = pd.read_parquet("./final_datasets/splitted_Datasets/training_Dataset/TrainingAnswers.parquet")

ids_review_text = df_train_inputs['id'].values
ids_labels = df_train_labels['id'].values

if not (ids_review_text == ids_labels).all():
    raise ValueError("Warning: Id's in review_body and sentiment are not alligned")

# Creating tensors , we transform the lists(series) in to matrizes 
input_ids = torch.tensor(df_train_inputs['tokenized_text'].to_list())
attention_mask = (input_ids !=0).long()                         # input_ids other than 0 are the attention mask 
labels = torch.tensor(df_train_labels['sentiment'].to_list())
row_id_tensor = torch.tensor(ids_review_text).long()


dataset_training = TensorDataset(
input_ids,  # Reviews the are the Questions for the model 
attention_mask, # attention mask , deciding what input-id's are important for the training , 0 get ignored they hold no inherent value
labels,  # answers for our input
row_id_tensor # original ids, these are not important for the training of the model but later for identifing the prediction of the model to the original id of the row 
)



train_loader = DataLoader(dataset_training, batch_size=16, shuffle = True) # Giving our model the training data in batch sizes, we can not add all the data at once 

optimizer = optimizer.AdamW(model.parameters(), lr=2e-5, eps =1e-8, weight_decay=0.01)  # Adjusting the learning rate and eps is a "Insurance" for the numerical stability this way the gradient will never get divided by zero, when the gradients get smaller

epochs = 5
total_training_volume = len(train_loader)* epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=150, num_training_steps = total_training_volume)

model.train()
for epoch in range(epochs):
    print(f"Start epoch {epoch +1} from {epochs}")
    total_loss = 0

    for batch in train_loader:               # The Model gets trained in batches of 16 
     input_ids = batch[0].to(device)         # Review Data(Questions) Transfer to Gpu
     attention_mask = batch[1].to(device)    # Attention mask data Transfer to the gpu 
     labels = batch[2].to(device)            # Answers to the gpu  
     
     model.zero_grad()                       # reseting the gradient, so the loss will not add up and grow to much. If the loss value is to high the adjusting value of the gradient will be to high and adjustments are to overpowered
     
     outputs = model(input_ids,
                     token_type_ids=None,
                     attention_mask=attention_mask,
                     labels=labels)
     
     loss = outputs.loss
     total_loss += loss.item()              # Calculating the loss per batch 
     
     loss.backward()                        # Calculating the error rate, how far was the predicted label from the real label? 
     
     optimizer.step()   # Adjusts the weights of just on the current batch, this is supported with the method zero_grad() that we applied to our model earlier in the loop
     scheduler.step() 
    average_train_loss = total_loss /len(train_loader)   # loss per badge 
    print(f"Average loss in epoch {epoch +1}: {average_train_loss:.4f}")  # if the train loss value per batch is sinking after every epoch, than this is a sign that the model learns


#Average loss in epoch 1: 1.0900
#Start epoch 2 from 3
#Average loss in epoch 2: 0.4439
#Start epoch 3 from 3
#Average loss in epoch 3: 0.2898


VERSION = "v6_lr2e-5_150warmupsteps_decay0.01_epochs5"  # ! dont forget to change name to desired name in the savedmodels Folder , needs to be done before training!
MODEL_SAVE_PATH = f"./saved_models/{VERSION}"

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model_to_save = model.module if hasattr(model, 'module') else  model 
model_to_save.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)