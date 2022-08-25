from dp_cgans import DP_CGAN, Onto_DP_CGAN, OntologyEmbedding
from datetime import datetime
import pandas as pd
import os


generated_files_path = '../persistent/model'
if not os.path.exists(generated_files_path):
    os.makedirs(generated_files_path)

tabular_data = pd.read_csv("../persistent/data/syn_data/syn_patients_data_seen_600_50_leukemia.csv", header=0)

print(f'Table data: {tabular_data}')

onto_embedding = OntologyEmbedding(embedding_path='../persistent/data/ontology/embeddings/hpObo_hoom_ordo_25_100s/ontology.embeddings',
                                   embedding_size=100,
                                   hp_dict_fn='../persistent/data/ontology/HPO.dict',
                                   rd_dict_fn='../persistent/data/ontology/ORDO.dict')

epochs = 10
model = Onto_DP_CGAN(
    embedding=onto_embedding,
    log_file_path=generated_files_path,
    epochs=epochs,
    batch_size=500,
    log_frequency=True,
    verbose=True,
    noise_dim=100,
    generator_dim=(128, 128, 128),
    discriminator_dim=(128, 128, 128),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    discriminator_steps=5,
    private=False,
)

print(f'Start model training, for {epochs} epochs')
model.fit(tabular_data)

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
print('Training finished, saving the model')
# Saving the model for future sampling
model.save(f'../persistent/model/{current_time}_{epochs}_epochs_onto_dp_cgans_model.pkl')

# Loading the list of unseen RDs for ZSL sampling
unseen_file = '../persistent/data/syn_data/unseen_rds_3_leukemia.txt'
picked_unseen_rds = []
with open(unseen_file) as uf:
    for rd in uf:
        picked_unseen_rds.append(rd.strip())

# Sample the generated synthetic data
nb_rows = 100
fn = os.path.join(generated_files_path, f'{current_time}_{epochs}_epochs_seen_sample_{nb_rows}_rows.csv')
print(f'Sampling {nb_rows} seen rows')
model.sample(nb_rows).to_csv(fn)

# ZSL sampling (unseen embeddings)
if len(picked_unseen_rds) > 0:
    fn = os.path.join(generated_files_path, f'{current_time}_{epochs}_epochs_unseen_sample_{nb_rows}_rows.csv')
    print(f'Sampling {nb_rows} unseen rows')
    model.sample(nb_rows, unseen_rds=picked_unseen_rds).to_csv(fn)