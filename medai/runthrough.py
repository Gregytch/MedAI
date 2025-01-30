from ml_logic.NLP import input_creator
from main import pred

#get data
with open("ml_logic/NLP_bio_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("ml_logic/dataset_col.pkl", "rb") as f:
    columns = pickle.load(f)


#get user input
text = input("Please enter all your current symptoms in plain text, seperated by a comma (,): ")

#do the NLP transformation
vector = input_creator(model, columns, text)

#Create a prediction
output = pred(vector)
