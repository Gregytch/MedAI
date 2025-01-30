from ml_logic.NLP import input_creator
from main import pred
import pickle
import os

##RELATIVE DIRECTORY
file=__file__
dir=os.path.dirname(file)
NLP_MODEL_PATH = os.path.join(dir, "ml_logic/NLP_bio_model.pkl")
COL_PATH = os.path.join(dir, "ml_logic/dataset_col.pkl")

#get data
with open(NLP_MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(COL_PATH, "rb") as f:
    columns = pickle.load(f)


#get user input
text = input("Please enter all your current symptoms in plain text, seperated by a comma (,): ")

#do the NLP transformation
vector = input_creator(model, columns, text)

#Create a prediction
output = pred(vector)
