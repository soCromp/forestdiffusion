import os
import pandas as pd 
import numpy as np 
from ForestDiffusion import ForestDiffusionModel
import json
import datetime 

out_path = '/home/sonia/ForestDiffusion/synth_data'
data_path = '/home/sonia/tabby/data'
datasets = ['abalone', 'diabetes-new', 'nouse-new', 'rain', 'travel', 'adult', 'glaucoma']

os.makedirs(out_path, exist_ok=True)

dataset = datasets[0]
df_train = pd.read_csv(os.path.join(data_path, dataset, 'latest/train.csv'))
np_train = df_train.to_numpy()
with open(os.path.join(data_path, dataset, 'latest/config.json')) as f:
    config = json.load(f)


# Sex,Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight,Class_number_of_rings
col_to_idx = {col: idx for idx, col in enumerate(df_train.columns)}

if config['task'] == 'classification':
    cat_indexes = [col_to_idx[col] for col in config['ords']+config['labs']]
else:
    cat_indexes = [col_to_idx[col] for col in config['ords']]
print('cat indexes', cat_indexes)
    
forest_model = ForestDiffusionModel(np_train, label_y=None, n_t=50, duplicate_K=100, remove_miss=False,
                                    bin_indexes=[], cat_indexes=cat_indexes, int_indexes=[], diffusion_type='flow', n_jobs=-1)
synth = forest_model.generate(batch_size=1000)#X.shape[0]) 

synthdf = pd.DataFrame(synth, columns=forest_model.X_names_after)
unassigned = {}
for col in set(forest_model.X_names_after) - set(forest_model.X_names_before):
    synthdf[col] = synthdf[col].round(0).astype(int)
    synthdf[synthdf[col] > 1][col] = 1
    synthdf[synthdf[col] < 0][col] = 0

postdummies = list(set(forest_model.X_names_after) - set(forest_model.X_names_before)) 
undummies = pd.from_dummies(synthdf[postdummies], sep='_')
synthdf = pd.concat([synthdf, undummies], axis=1)
synthdf = synthdf.drop(columns=postdummies)
synthdf = synthdf[forest_model.X_names_before]
synthdf.columns = df_train.columns

now = datetime.datetime.now().strftime("%I.%M%p.%B.%d")
synthdf.to_csv(os.path.join(out_path, f'{dataset}_{now}.csv'), index=False)

print(synthdf)
