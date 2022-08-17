from sdv import SDV
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

nb_rows = 100
model_file = '../persistent/model/2022_08_15_20_52_05_onto_dp_cgans_model.pkl'
save_path = f'../persistent/model/{current_time}_generated_{nb_rows}_rows.csv'

model = SDV.load(model_file)
syn_data = model.sample(nb_rows).to_csv(save_path)