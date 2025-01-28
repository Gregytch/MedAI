import numpy as np
import pandas as pd


from models.data import clean_data

def preprocess():

    # Load data from directory (relatve path)
    df_symp = pd.read_csv('/home/greg_ytch/code/Gregytch/MedAI/raw_data/Final_Augmented_dataset_Diseases_and_Symptoms.csv')

    #Clean the data calling the function clean_data
    data=clean_data(df_symp)

    print(data.shape)

    return data

if __name__ == '__main__':
    preprocess()
