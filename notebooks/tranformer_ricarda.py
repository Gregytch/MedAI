import os
import pickle

dir=os.path.dirname(__file__)
NLP_MODEL_PATH = os.path.join(dir, "../models/NLP_bio_model.pkl")
COL_PATH = os.path.join(dir, "../models/dataset_col.pkl")

#get data
with open(NLP_MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(COL_PATH, "rb") as f:
    columns = pickle.load(f)

embeddings_columns = model.encode(columns)

with open(os.path.join(dir,"../models/embeddings_columns.pkl"), "wb") as f:
        pickle.dump(embeddings_columns, f)
