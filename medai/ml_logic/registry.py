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
    file=__file__
    dir=os.path.dirname(file)
    MODEL_PATH = os.path.join(dir, "model_rf.pkl")
    ENCODER_PATH = os.path.join(dir, "label_encoder.pkl")

    ## Load model from directory (works when same directory, otherwise add relative phat above path)

    with open(MODEL_PATH, 'rb') as rf:
        model = pickle.load(rf)


    ##Load Encoder used with model
    with open(ENCODER_PATH, "rb") as le:
        label_encoder = pickle.load(le)

        return model, label_encoder
