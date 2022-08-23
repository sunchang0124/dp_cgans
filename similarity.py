from sdv.evaluation import evaluate
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Generated from the ontology
fake_seen_patients_fn = '../persistent/data/syn_data/syn_patients_data_seen_100_50.csv'
fake_unseen_patients_fn = '../persistent/data/syn_data/syn_patients_data_unseen_100_353.csv'

# Sampled from the Onto_CGAN model
sampled_seen_patients_fn = '../persistent/model/2022_08_23_08_01_52_seen_final_sample_100_rows.csv'
sampled_unseen_patients_fn = '../persistent/model/2022_08_22_19_32_55_unseen_final_sample_100_rows.csv'

sampled_seen_patients = pd.read_csv(sampled_seen_patients_fn)
sampled_seen_patients.drop(columns=sampled_seen_patients.columns[0], axis=1, inplace=True)
sampled_unseen_patients = pd.read_csv(sampled_unseen_patients_fn)
sampled_unseen_patients.drop(columns=sampled_unseen_patients.columns[0], axis=1, inplace=True)
fake_seen_patients = pd.read_csv(fake_seen_patients_fn)
fake_unseen_patients = pd.read_csv(fake_unseen_patients_fn)
fake_unseen_patients = fake_unseen_patients[sampled_unseen_patients.columns]

print(f'Fake Seen Patients: {len(fake_seen_patients)}x{len(fake_seen_patients.columns)} Fake Unseen Patients: {len(fake_unseen_patients)}x{len(fake_unseen_patients.columns)} Sampled Seen Patients: {len(sampled_seen_patients)}x{len(sampled_seen_patients.columns)} Sampled Unseen Patients: {len(sampled_unseen_patients)}x{len(sampled_unseen_patients.columns)}')

print(f'Similarity between fake and sampled seen patients data: {evaluate(fake_seen_patients, sampled_seen_patients)}')
print(f'Similarity between fake and sampled unseen patients data: {evaluate(fake_unseen_patients, sampled_unseen_patients)}')