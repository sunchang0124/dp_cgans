from gensim.models import KeyedVectors
import numpy as np


class OntologyEmbedding():

    def __init__(self, embedding_path, embedding_size, hp_dict_fn, rd_dict_fn):
        self._embedding_model = KeyedVectors.load(embedding_path)
        self._embedding_size = embedding_size
        self._iri_dict = {}

        with open(hp_dict_fn) as f:
            for line in f:
                (entity, iri) = line.strip().split(';')
                self._iri_dict[entity] = iri

        with open(rd_dict_fn) as f:
            for line in f:
                (entity, iri) = line.strip().split(';')
                self._iri_dict[entity] = iri

    def get_iri(self, entity):
        return self._iri_dict.get(entity, None)

    def get_embedding(self, entity):
        print(f'Retrieving embedding for: {entity}')
        iri = self.get_iri(entity)
        if iri is not None:
            return self._embedding_model.wv[iri]
        return np.zeros(self._embedding_size)

    def get_embedding_size(self):
        return self._embedding_size
