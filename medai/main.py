import numpy as np
import pandas as pd
import os
import pickle

from ml_logic.data import clean_data
from ml_logic.NLP import input_creator
from ml_logic.registry import load_model


def preprocess():
    print( "\n⚙️ Preprocessing data" )

    dir=os.path.dirname(__file__)


    # Load data from directory (relative path)
    df_symp = pd.read_csv(os.path.join(dir,'../raw_data/Final_Augmented_dataset_Diseases_and_Symptoms.csv'))


    #Clean the data calling the function clean_data
    data=clean_data(df_symp)

    #Creating X and y
    X=data.drop(['diseases'], axis=1)
    y=data['diseases']
    columns = list(X.columns)



    with open(os.path.join(dir,"../models/dataset_col.pkl"), "wb") as f:
        pickle.dump(columns, f)

    #Print the shape of the dataset, X and y
    print(f"Shape of the dataset : {data.shape}")
    print(f"Shape of the features X (Symptoms): {X.shape}")
    print(f"Shape of the target y (Diseases): {y.shape}")

    #Later -> store in BQ

    return data, X, y

def pred(X_pred) :

    print( "\n🔮 Predicting diseases" )


    #load the model
    model, label_encoder = load_model()

    #predict the most likely diseases
    probs=model.predict_proba(X_pred)[0]
    print("\n🧮 Probabilities computed")
    #create dataframe with results
    df_probs = pd.DataFrame({
        "Disease": label_encoder.classes_,  # List of disease names
        "Probability": probs   # Corresponding probabilities
    })

    #sorting by highest proba
    df_probs_sorted = df_probs.sort_values(by="Probability", ascending=False).reset_index(drop=True)

    #print results
    print(f"✅ pred() done")
    print(f"🏥 To ten predicted disease with probability:\n {df_probs_sorted[0:10]}")

    return df_probs_sorted


def runthough():

    print( "\n🏃 Starting runthrough" )
    ##RELATIVE DIRECTORY
    dir=os.path.dirname(__file__)
    NLP_MODEL_PATH = os.path.join(dir, "../models/NLP_bio_model.pkl")
    COL_PATH = os.path.join(dir, "../models/dataset_col2.pkl")

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
    return output

def runthrough_api(user_input):

    print( "\n🏃 Starting runthrough" )
    ##RELATIVE DIRECTORY
    dir=os.path.dirname(__file__)
    NLP_MODEL_PATH = os.path.join(dir, "../models/NLP_bio_model.pkl")
    COL_PATH = os.path.join(dir, "../models/dataset_col2.pkl")

    #get data
    with open(NLP_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(COL_PATH, "rb") as f:
        columns = pickle.load(f)

    #do the NLP transformation
    vector = input_creator(model, columns, user_input)

    #Create a prediction
    output = pred(vector)
    return output

#When we run main.py Will instanciate all code but run only what is under if __name__ == '__main__':
if __name__ == '__main__':
    #preprocess()
    runthough()
