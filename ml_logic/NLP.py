from sentence_transformers import SentenceTransformer, util
import numpy as np

def input_creator(text, data):
    model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
    #Load Model from pickle
    #model = load_nlp_model()

    #prepare
    data = data
    columns = list(data.columns)

    #tokenizer
    symptoms = text.split(",")
    symptoms = [a.strip() for a in symptoms]

    #embeddings
    embeddings_symptoms = model.encode(symptoms)
    embeddings_columns = model.encode(columns)

    #similarities
    similiarities = np.zeros(shape = (len(columns), len(symptoms)))

    # Compute cosine similarity
    for i in range(len(symptoms)):
        cosine_scores = util.cos_sim(embeddings_symptoms[i], embeddings_columns)
        print(f"{symptoms[i]} matches {columns[np.argmax(cosine_scores)]} with probability {cosine_scores.max()}")
