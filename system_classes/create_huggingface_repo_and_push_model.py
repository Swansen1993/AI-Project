from huggingface_hub import create_repo
from huggingface_hub import login
from huggingface_hub import upload_folder

login()


MODEL_PATH = "v6_lr2e-5_150warmupsteps_decay0.01_epochs5"       # ! folder name check



# Model upload 
upload_folder(folder_path=f"./saved_models/{MODEL_PATH}",
              repo_id = "Swansen1993/RepositoryBertModels",
              path_in_repo=f"Swansen1993/RepositoryBertModels/{MODEL_PATH}",
              repo_type = "model")

