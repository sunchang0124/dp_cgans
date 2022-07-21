from dp_cgans import DP_CGAN, Onto_DP_CGAN
import pandas as pd
import os

result_samples_path = '../persistent/data/gan_syn_data/'
if not os.path.exists(result_samples_path):
    os.makedirs(result_samples_path)

tabular_data = pd.read_csv("../persistent/data/syn_data/small_syn_patients_data_seen.csv", header=0)

# We adjusted the original CTGAN model from SDV. Instead of looking at the distribution of individual variable, we extended to two variables and keep their corrll
model = Onto_DP_CGAN(
    embeddings_fn='../persistent/data/ontology/embeddings/Onto_TransE.pkl',
    sample_epochs=100,
    sample_epochs_path=result_samples_path,
    primary_key='patient_id',
    epochs=1, # number of training epochs
    batch_size=1000, # the size of each batch
    log_frequency=True,
    verbose=True,
    embedding_dim=100,
    generator_dim=(128, 128, 128),
    discriminator_dim=(128, 128, 128),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    discriminator_steps=1,
    private=False,
)

print("Start model training")
model.fit(tabular_data)

# Sample the generated synthetic data
nb_rows = 100
model.sample(nb_rows).to_csv(os.path.join(result_samples_path, f'final_sample_{nb_rows}.csv')
