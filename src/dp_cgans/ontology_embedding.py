from gensim.models import KeyedVectors


class OntologyEmbedding():

    def __init__(self, embedding_path, hp_dict_fn, rd_dict_fn):
        # load embeddings here
        self._embedding_model = KeyedVectors.load(embedding_path, mmap='r')
        self._hp_dict = {}
        self._rd_dict = {}

        with open(hp_dict_fn) as f:
            for line in f:
                (name, iri) = line.split(',')
                self._hp_dict[name] = iri

        with open(rd_dict_fn) as f:
            for line in f:
                (name, iri) = line.split(',')
                self._rd_dict[name] = iri

    def get_rd_iri(self, rd_name):
        return self._rd_dict[rd_name]

    def get_hp_iri(self, hp_name):
        return self._hp_dict[hp_name]

    def get_embedding(self, term):
        return self._embedding_model.wv[term]
