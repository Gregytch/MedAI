import pandas as pd
import os
import pickle

from medai.ml_logic.NLP import input_creator
from medai.ml_logic.registry import load_model
from medai.ml_logic.registry import load_symptoms



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
    print(f"🏥 Top 5 predicted disease with probability:\n {df_probs_sorted[0:5]}")


    return df_probs_sorted[0:5]




def runthrough_api(user_input):

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

    #do the NLP transformation
    vector, symptoms_to_use = input_creator(model, columns, user_input)

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


    #Add the list of diseases and symptom used to make it to the output dictionary
    output_dict["Predictions"] = predictions
    output_dict["Used_Symptoms"] = symptoms_to_use

    print(f"🏁 runthrough_api() done")
    return output_dict

#When we run main.py Will instanciate all code but run only what is under if __name__ == '__main__':
if __name__ == '__main__':
    runthrough_api()
