from sdv import SDV
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

nb_rows = 600
model_file = '../persistent/model/leukemia_600_50_5000_epochs_onto_dp_cgans_model.pkl'
seen_save_path = f'../persistent/model/{current_time}_seen_sample_{nb_rows}_rows.csv'
unseen_save_path = f'../persistent/model/{current_time}_unseen_sample_{nb_rows}_rows.csv'

model = SDV.load(model_file)

# Unseen ZSL sampling
unseen_file = '../persistent/data/syn_data/unseen_rds_3_leukemia.txt'
picked_unseen_rds = []
with open(unseen_file) as uf:
    for rd in uf:
        picked_unseen_rds.append(rd.strip())

print(f'Sampling {nb_rows} seen rows')
model.sample(nb_rows).to_csv(seen_save_path)
print(f'Sampling {nb_rows} unseen rows')
model.sample(nb_rows, unseen_rds=picked_unseen_rds).to_csv(unseen_save_path)