# -*- coding: utf-8 -*-

# configure logging for the library with a null handler (nothing is printed by default). See
# http://docs.pthon-guide.org/en/latest/writing/logging/

__author__ = 'Chang Sun'
__email__ = 'chang.sun@maastrichtuniversity.nl'
__version__ = '0.1.0'


# from dp_cgans import constraints, metadata
from dp_cgans.synthesizers.dp_cgan_modular import DP_CGAN
from dp_cgans.synthesizers.dp_cgan import DPCGANSynthesizer
from dp_cgans.Transformers.transformers import OneHotEncoder

from dp_cgans.ontocgan.onto_cgan_modular import Onto_DP_CGAN
from dp_cgans.ontocgan.onto_cgan import Onto_DPCGANSynthesizer
from dp_cgans.ontocgan.ontology_embedding import OntologyEmbedding

__all__ = (
    'DP_CGAN',
    'DPCGANSynthesizer',
    'OneHotEncoder',
    'Onto_DPCGANSynthesizer',
    'OntologyEmbedding',
    'Onto_DP_CGAN'
)
