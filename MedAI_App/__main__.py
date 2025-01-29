import numpy as np
import pandas as pd


from models.data import clean_data

def preprocess():

    # Load data from directory (relatve path)
    df_symp = pd.read_csv('/home/greg_ytch/code/Gregytch/MedAI/raw_data/Final_Augmented_dataset_Diseases_and_Symptoms.csv')

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

if __name__ == '__main__':
    preprocess()
