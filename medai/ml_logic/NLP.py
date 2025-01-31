import numpy as np
import pandas as pd
from sentence_transformers import util
from symspellpy import SymSpell
import os
import pickle

# #FOR TESTING
# dir=os.path.dirname(__file__)
# NLP_MODEL_PATH = os.path.join(dir, "..", "..", "models", "NLP_bio_model.pkl")
# COL_PATH = os.path.join(dir, "..", "..", "models", "dataset_col.pkl")
# print(NLP_MODEL_PATH)
# #get data
# with open(NLP_MODEL_PATH, "rb") as f:
#     model = pickle.load(f)
# with open(COL_PATH, "rb") as f:
#     columns = pickle.load(f)


#SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
def input_creator(model, columns, text):
    #tokenizer
    symptoms = text.split(",")
    symptoms = [a.strip() for a in symptoms]

    # Initialize SymSpell
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    dictionary_path = os.path.join(dir, "..", "..", "models", "frequency_dictionary_en_82_765.txt")
    # Load a frequency dictionary
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    # Correct typos
    for index, element in enumerate(symptoms):
        suggestion = sym_spell.lookup_compound(element, max_edit_distance=2)
        symptoms[index] = (suggestion[0].term)


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
        if cosine_scores.max() >= 0.6:
            vector[columns[np.argmax(cosine_scores)]] = 1
    print("-------- Done---------------")
    return vector


# text1 = "headache, bad fever, rash, nauseous, pressure on the eye"
# input_creator(model, columns,text1)
# #
# text2 = "head ache, headache, hypertenssion,eadache"
# input_creator(model, columns,text2)
#
#text3 = "red eye, very high fever, fruit cake, swollen leg"
#input_creator(model, columns,text3)
