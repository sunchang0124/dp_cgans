from sdv.evaluation import evaluate
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


def print_similarity(title, fsp_fn, fup_fn, ssp_fn, sup_fn):
    sampled_seen_patients = pd.read_csv(ssp_fn)
    sampled_seen_patients.drop(columns=sampled_seen_patients.columns[0], axis=1, inplace=True)
    sampled_unseen_patients = pd.read_csv(sup_fn)
    sampled_unseen_patients.drop(columns=sampled_unseen_patients.columns[0], axis=1, inplace=True)
    fake_seen_patients = pd.read_csv(fsp_fn)
    fake_unseen_patients = pd.read_csv(fup_fn)
    fake_unseen_patients = fake_unseen_patients[sampled_unseen_patients.columns]

    print(f'Results for {title}')
    print(f'Fake Seen Patients: {len(fake_seen_patients)}x{len(fake_seen_patients.columns)} Fake Unseen Patients: {len(fake_unseen_patients)}x{len(fake_unseen_patients.columns)} Sampled Seen Patients: {len(sampled_seen_patients)}x{len(sampled_seen_patients.columns)} Sampled Unseen Patients: {len(sampled_unseen_patients)}x{len(sampled_unseen_patients.columns)}')
    print(f'Similarity between fake and sampled seen patients data: {evaluate(fake_seen_patients, sampled_seen_patients)}')
    print(f'Similarity between fake and sampled unseen patients data: {evaluate(fake_unseen_patients, sampled_unseen_patients)}\n')


print_similarity(title='Brain and lung RDs, 100x50 dataset, 5000 epochs',
                 fsp_fn='../persistent/data/syn_data/syn_patients_data_seen_100_50.csv',
                 fup_fn='../persistent/data/syn_data/syn_patients_data_unseen_100_353.csv',
                 ssp_fn='../persistent/model/brain_lung_100_50_seen_sample_100_rows.csv',
                 sup_fn='../persistent/model/brain_lung_100_50_unseen_sample_100_rows.csv'
                )

print_similarity(title='Leukemia RDs, 600x50 dataset, 5000 epochs',
                 fsp_fn='../persistent/data/syn_data/syn_patients_data_seen_600_50_leukemia.csv',
                 fup_fn='../persistent/data/syn_data/syn_patients_data_unseen_150_50_leukemia.csv',
                 ssp_fn='../persistent/model/2022_08_23_13_23_06_seen_sample_600_rows.csv',
                 sup_fn='../persistent/model/2022_08_23_13_23_06_unseen_sample_600_rows.csv'
                )