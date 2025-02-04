import numpy as np
import pandas as pd
from sentence_transformers import util
from symspellpy import SymSpell
import os
import pickle


#SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
def input_creator(model, columns, text):
    #tokenizer
    symptoms = text.split(",")
    symptoms = [a.strip() for a in symptoms]

    # Initialize SymSpell
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    dir=os.path.dirname(__file__)
    dictionary_path = os.path.join(dir, "..", "..", "models", "frequency_dictionary_en_82_765.txt")
    # Load a frequency dictionary
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    # Correct typos
    for index, element in enumerate(symptoms):
        suggestion = sym_spell.lookup_compound(element, max_edit_distance=2)
        symptoms[index] = (suggestion[0].term)

    dir = dir=os.path.dirname(__file__)
    print(dir)
    #embeddings
    if os.path.exists(os.path.join(dir,"../../models/embeddings_columns.pkl")):
        with open(os.path.join(dir,"../../models/embeddings_columns.pkl"), "rb") as f:
            embeddings_columns = pickle.load(f)
        print("File with embeddings already there")
    else:
        embeddings_columns = model.encode(columns)
        with open(os.path.join(dir,"../../models/embeddings_columns.pkl"), "wb") as f:
            pickle.dump(embeddings_columns, f)
        print("File with embeddings created newly")

    embeddings_symptoms = model.encode(symptoms)

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
