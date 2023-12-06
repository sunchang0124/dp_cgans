from gensim.models import KeyedVectors
import numpy as np


class OntologyEmbedding():
    """Ontology embedding class.

    This class is specifically designed for embeddings with the RDs + HPs datasets.

    Args:
    embedding_path (str):
        Path to the embeddings file, to be loaded with KeyedVectors (.embedding file).
    embedding_size (int):
        Size of each embedding vector.
    hp_dict_fn (str):
        Path to the HP label -> HP URI dictionary file.
    rd_dict_fn (str):
        Path to the RD label -> RD URI dictionary file.
    """

    def __init__(self, embedding_path, embedding_size, hp_dict_fn, rd_dict_fn):
        self._embed_model = KeyedVectors.load(embedding_path)
        self.embed_size = embedding_size
        self._iri_dict = {}

        with open(hp_dict_fn) as f:
            for line in f:
                (entity, iri) = line.strip().split(';')
                self._iri_dict[entity] = iri

        with open(rd_dict_fn) as f:
            for line in f:
                (entity, iri) = line.strip().split(';')
                self._iri_dict[entity] = iri

    # def get_iri(self, entity):
    #     """Get IRI of entity from its label.

    #     Args:
    #         entity (str):
    #             Label of the class to get IRI of.
    #     Returns:
    #         (str/None): IRI of the corresponding entity, None if not found in the dictionary.
    #     """
    #     return self._iri_dict.get(entity, None)

    def get_embedding(self, entity):
        """Get embedding of entity.

        Args:
            entity (str):
                Label or URI of the class to get the embedding of.
        Returns:
            (numpy.ndarray): Embedding of the entity, array full of zeros if embedding not found.
        """
        # iri = self.get_iri(entity)
        # if iri is not None:
        #     return self._embed_model.wv[iri]

        # print("http://www.orpha.net/ORDO/Orphanet_"+str(int(entity)))

        # try: 
        return self._embed_model.wv[entity]#["http://www.orpha.net/ORDO/Orphanet_"+str(int(entity))]
        # except:
        #     print("The disease iri %s is not found in the embeddings." %entity)
        #     return np.zeros(self.embed_size)
