from dp_cgans import DP_CGAN, Onto_DP_CGAN, OntologyEmbedding
from datetime import datetime
import pandas as pd
import os


result_samples_path = '../persistent/model'
if not os.path.exists(result_samples_path):
    os.makedirs(result_samples_path)

tabular_data = pd.read_csv("../persistent/data/syn_data/syn_patients_data_seen_600_50_leukemia.csv", header=0)

print(f'Table data: {tabular_data}')

columns = tabular_data.columns.values.tolist()

onto_embedding = OntologyEmbedding(embedding_path='../persistent/data/ontology/embeddings/hpObo_hoom_ordo_25_100s/ontology.embeddings',
                                   embedding_size=100,
                                   hp_dict_fn='../persistent/data/ontology/HPO.dict',
                                   rd_dict_fn='../persistent/data/ontology/ORDO.dict')  # Embeddings_number at 1 means just RD embedding

# We adjusted the original CTGAN model from SDV. Instead of looking at the distribution of individual variable, we extended to two variables and keep their corrll
epochs = 10
model = Onto_DP_CGAN(
    embedding=onto_embedding,
    columns=columns,
    log_file_path=result_samples_path,
    epochs=epochs, # number of training epochs
    batch_size=500, # the size of each batch
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
model.save(f'../persistent/model/{current_time}_{epochs}_epochs_onto_dp_cgans_model.pkl')

# Unseen ZSL sampling
unseen_file = '../persistent/data/syn_data/unseen_rds_3_leukemia.txt'
picked_unseen_rds = []
with open(unseen_file) as uf:
    for rd in uf:
        picked_unseen_rds.append(rd.strip())

# Sample the generated synthetic data
nb_rows = 100
fn = os.path.join(result_samples_path, f'{current_time}_{epochs}_epochs_seen_sample_{nb_rows}_rows.csv')
print(f'Sampling {nb_rows} seen rows')
model.sample(nb_rows).to_csv(fn)

if len(picked_unseen_rds) > 0:
    fn = os.path.join(result_samples_path, f'{current_time}_{epochs}_epochs_unseen_sample_{nb_rows}_rows.csv')
    print(f'Sampling {nb_rows} unseen rows')
    model.sample(nb_rows, unseen_rds=picked_unseen_rds).to_csv(fn)