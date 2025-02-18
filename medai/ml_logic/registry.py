import os
import time
import pickle

def load_model(stage="Production"):
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    print("\n💾 Load latest model from local registry..." )

    ##RELATIVE DIRECTORY
    dir=os.path.dirname(__file__)
    MODEL_PATH = os.path.join(dir, "../../models/xgb_model_full.pkl")
    ENCODER_PATH = os.path.join(dir, "../../models/label_encoder.pkl")

    ## Load model from directory

    with open(MODEL_PATH, 'rb') as xgb:
        model = pickle.load(xgb)
    print("\n💾 Model loaded")

    ##Load Encoder used with model
    with open(ENCODER_PATH, "rb") as le:
        label_encoder = pickle.load(le)
        print("\n💾 Encoder loaded")
        return model, label_encoder

def load_symptoms():
    """
    Load symptoms from local registry
    """

    print("\n💾 Load symptoms from local registry..." )

    ##RELATIVE DIRECTORY
    dir=os.path.dirname(__file__)
    SYMPTOMS_PATH = os.path.join(dir, "../../models/disease_symptom_dict.pkl")

    ## Load symptoms from directory

    with open(SYMPTOMS_PATH, 'rb') as symp:
        symptoms = pickle.load(symp)
    print("\n💾 Symptoms loaded")

    return symptoms
