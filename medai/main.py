import numpy as np
import pandas as pd
import os

from ml_logic.data import clean_data

from ml_logic.registry import load_model
from sklearn.preprocessing import LabelEncoder


def preprocess():
    file=__file__
    dir=os.path.dirname(file)
    # Load data from directory (relative path)
    df_symp = pd.read_csv(os.path.join(dir,'../raw_data/Final_Augmented_dataset_Diseases_and_Symptoms.csv'))

    #Clean the data calling the function clean_data
    data=clean_data(df_symp)

    #Creating X and y
    X=data.drop(['diseases'], axis=1)
    y=data['diseases']


    #Print the shape of the dataset, X and y
    print(f"Shape of the dataset : {data.shape}")
    print(f"Shape of the features X (Symptoms): {X.shape}")
    print(f"Shape of the target y (Diseases): {y.shape}")

    #Later -> store in BQ

    return data, X, y

def pred(X_pred) :

    print( "\n ⭐️ predicting diseases" )


    #load the model
    model, label_encoder = load_model()

    #predict the most likely diseases
    probs=model.predict_proba(X_pred)[0]

    #create dataframe with results
    df_probs = pd.DataFrame({
        "Disease": label_encoder.classes_,  # List of disease names
        "Probability": probs   # Corresponding probabilities
    })

    #sorting by highest proba
    df_probs_sorted = df_probs.sort_values(by="Probability", ascending=False).reset_index(drop=True)

    #print results
    print(f"✅ pred() done")
    print(f"🏥 To ten predicted disease with probability: {df_probs_sorted[0:10]}")

    return df_probs_sorted



if __name__ == '__main__':
    data,X,y=preprocess()
    pred(X[0:1])
