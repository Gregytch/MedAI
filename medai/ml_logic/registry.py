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
    MODEL_PATH = os.path.join(dir, "../../models/XGB_model.pkl")
    ENCODER_PATH = os.path.join(dir, "../../models/label_encoder.pkl")

    ## Load model from directory (works when same directory, otherwise add relative phat above path)

    with open(MODEL_PATH, 'rb') as rf:
        model = pickle.load(rf)
    print("\n💾 Model loaded")

    ##Load Encoder used with model
    with open(ENCODER_PATH, "rb") as le:
        label_encoder = pickle.load(le)
        print("\n💾 Encoder loaded")
        return model, label_encoder
