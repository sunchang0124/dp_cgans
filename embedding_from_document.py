import multiprocessing
import configparser
import click
import logging
import os
import time
import gensim


def extract_owl2vec_model(config_file, document_file):
    config = configparser.ConfigParser()
    config.read(click.format_filename(config_file))

    if 'cache_dir' not in config['DOCUMENT']:
        config['DOCUMENT']['cache_dir'] = './cache'

    if not os.path.exists(config['DOCUMENT']['cache_dir']):
        os.mkdir(config['DOCUMENT']['cache_dir'])

    model_ = __perform_ontology_embedding(config, document_file)

    return model_


def __perform_ontology_embedding(config, document):

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    all_doc = []
    with open(document, 'r') as sentences:
        for s in sentences:
            all_doc.append(s)

    # learn the language model (train a new model or fine tune the pre-trained model)
    start_time = time.time()
    if 'pre_train_model' not in config['MODEL'] or not os.path.exists(config['MODEL']['pre_train_model']):
        logging.info('Train the language model ...')
        model_ = gensim.models.Word2Vec(all_doc, vector_size=int(config['MODEL']['embed_size']),
                                        window=int(config['MODEL']['window']),
                                        workers=multiprocessing.cpu_count(),
                                        sg=1, epochs=int(config['MODEL']['iteration']),
                                        negative=int(config['MODEL']['negative']),
                                        min_count=int(config['MODEL']['min_count']), seed=int(config['MODEL']['seed']))
    else:
        logging.info('Fine-tune the pre-trained language model ...')
        model_ = gensim.models.Word2Vec.load(config['MODEL']['pre_train_model'])
        if len(all_doc) > 0:
            model_.min_count = int(config['MODEL']['min_count'])
            model_.build_vocab(all_doc, update=True)
            model_.train(all_doc, total_examples=model_.corpus_count, epochs=int(config['MODEL']['epoch']))

    logging.info('Time for learning the language model: %s seconds' % (time.time() - start_time))

    return model_


config_file = './embedding.cfg'
document_file = '../persistent/data/ontology/embeddings/hpObo_hoom_ordo_25_10s/document_sentences.txt'
gensim_model = extract_owl2vec_model(config_file, document_file)

# Gensim format
gensim_model.save(os.path.join('../persistent/data/ontology/embeddings/hpObo_hoom_ordo_25_10s/', 'ontology.embeddings'))