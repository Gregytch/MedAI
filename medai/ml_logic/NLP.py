import numpy as np
import pandas as pd
import pickle
from sentence_transformers import util

with open("NLP_bio_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("dataset_col.pkl", "rb") as f:
    columns = pickle.load(f)


#SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
def input_creator(model, columns, text):
    #tokenizer
    symptoms = text.split(",")
    symptoms = [a.strip() for a in symptoms]

    #embeddings
    embeddings_symptoms = model.encode(symptoms)
    embeddings_columns = model.encode(columns)

    #Setup output file
    zero_data = np.zeros(shape=(1, len(columns)))
    vector = pd.DataFrame(zero_data, columns=columns)

    # Compute cosine similarity
    for i in range(len(symptoms)):
        cosine_scores = util.cos_sim(embeddings_symptoms[i], embeddings_columns)
        print(f"{symptoms[i]} matches {columns[np.argmax(cosine_scores)]} with probability {cosine_scores.max()}")
        if cosine_scores.max() > 0.6:
            vector[columns[np.argmax(cosine_scores)]] = 1
    print("-------- Done---------------")
    return vector


text1 = "headache, bad fever, rash, nauseous, pressure on the eye"
input_creator(model, columns,text1)

text2 = "heartburn, tight chest, shaking, nervousness, panic"
input_creator(model, columns,text2)

text3 = "red eye, very high fever, fruit cake, swollen leg"
input_creator(model, columns,text3)
