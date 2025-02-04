import pandas as pd
import os
import pickle

from medai.ml_logic.data import clean_data
from medai.ml_logic.NLP import input_creator
from medai.ml_logic.registry import load_symptoms
from medai.main import pred

def preprocess():
    print( "\n⚙️ Cleaning data" )

    dir=os.path.dirname(__file__)


    # Load data from directory (relative path)
    df_symp = pd.read_csv(os.path.join(dir,'../raw_data/Final_Augmented_dataset_Diseases_and_Symptoms.csv'))


    #Clean the data calling the function clean_data
    data=clean_data(df_symp)

    #Creating X and y
    X=data.drop(['diseases'], axis=1)
    y=data['diseases']
    columns = list(X.columns)

    print(f"✅ Dataset cleaned")
    #Print the shape of the dataset, X and y
    print(f"  --Shape of the dataset : {data.shape}")
    print(f"  --Shape of the features X (Symptoms): {X.shape}")
    print(f"  --Shape of the target y (Diseases): {y.shape}")
    print(f"  --Shape of the target y (Diseases): {y.shape}")

    #Save the columns
    with open(os.path.join(dir,"../models/dataset_col.pkl"), "wb") as f:
        pickle.dump(columns, f)

    print(f"💾 Dataset columns exported in {os.path.join(dir,'/models/dataset_col.pkl')}")

    print( "\n⚙️ Computing symptoms' weights" )
    # Initialize dictionary of diseases and their symptoms with weights
    disease_symptom_dict = {}

    # Iterate through each disease
    for disease in data["diseases"].unique():
        #Select all rows related to this
        disease_rows = data[data["diseases"] == disease].drop(columns=["diseases"])

        # Count occurrences of each symptom
        symptom_counts = disease_rows.sum()

        # Total number of observations for this disease
        total_cases = len(disease_rows)

        # Compute weight = (occurrences / total cases)
        symptom_weights = (symptom_counts / total_cases)

        # Remove symptoms with 0 occurrences
        symptom_weights = symptom_weights[symptom_weights > 0].to_dict()

        # Store in dictionary
        disease_symptom_dict[disease] = symptom_weights

    #Sort the dictionary by symptoms' weights
    for disease in disease_symptom_dict:
        disease_symptom_dict[disease] = {k: v for k, v in sorted(disease_symptom_dict[disease].items(), key=lambda item: item[1], reverse=True)}


    print(f"  --✅ Computed weights for {len(disease_symptom_dict)} diseases")

    #Save the dictionary

    with open(os.path.join(dir,"../models/disease_symptom_dict.pkl"), "wb") as f:
        pickle.dump(disease_symptom_dict, f)

    print(f"💾 Disease x symptom weights dictionary saved in {os.path.join(dir,'/models/disease_symptom_dict.pkl')}")



def runthough():

    print( "\n🏃 Starting runthrough" )
    ##RELATIVE DIRECTORY
    dir=os.path.dirname(__file__)
    NLP_MODEL_PATH = os.path.join(dir, "../models/NLP_bio_model.pkl")
    COL_PATH = os.path.join(dir, "../models/dataset_col.pkl")

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


    #Convert pred output to list of dictionaries
    output_dict = {}
    predictions = []

        ##Create a list of dictionaries of diseases and their probabilities
    for i in range(len(output)):
        predictions.append(output.iloc[i].to_dict())

    #Add the dictionary of symptoms to the predictions list

        ## Load disease_symptom_dict
    disease_symptom_dict=load_symptoms()

        ##Add symptoms to the list

    for i in range(len(predictions)):
        predictions[i]['Symptoms'] = disease_symptom_dict[predictions[i]['Disease']]


    #Add the list to the output dictionary
    output_dict["Predictions"] = predictions

    return output_dict
