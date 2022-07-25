import nltk
import os
from owl2vec_star import owl2vec_star

nltk.download('punkt')

output_folder = '../persistent/data/ontology/embeddings/hpObo_hoom_ordo'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

"""
Parameters:
    ontology_file
    config_file
    uri_doc
    lit_doc
    mix_doc
    -> modify the cfg file for more params (cache dir, epochs, etc.)
"""
gensim_model = owl2vec_star.extract_owl2vec_model(None, "./embedding.cfg", True, True, True)

# Gensim format
gensim_model.save(os.path.join(output_folder, 'ontology.embeddings'))

# Text format (not required)
gensim_model.wv.save_word2vec_format(os.path.join(output_folder, "ontology.embeddings.txt"), binary=False)