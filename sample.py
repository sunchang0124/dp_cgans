from sdv import SDV
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

nb_rows = 100
model_file = '../persistent/model/2022_08_22_19_32_55_onto_dp_cgans_model.pkl'
save_path = f'../persistent/model/{current_time}_seen_final_sample_{nb_rows}_rows.csv'

model = SDV.load(model_file)
syn_data = model.sample(nb_rows).to_csv(save_path)

print(f'Sampling {nb_rows} seen rows')
model.sample(nb_rows).to_csv(save_path)