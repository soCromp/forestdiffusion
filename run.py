import os
import pandas as pd 
import numpy as np 
from ForestDiffusion import ForestDiffusionModel
import json
import datetime 

n_samples = 10
out_path = '/home/sonia/ForestDiffusion/synth_data'
data_path = '/home/sonia/tabby/data'
datasets = ['abalone', 'diabetes-new', 'house-new', 'rain', 'travel', 'adult', 'glaucoma']

os.makedirs(out_path, exist_ok=True)
now = datetime.datetime.now().strftime("%I.%M%p.%B.%d")

for dataset in datasets:
    df_train = pd.read_csv(os.path.join(data_path, dataset, 'latest/val.csv'))
    np_train = df_train.to_numpy()
    with open(os.path.join(data_path, dataset, 'latest/config.json')) as f:
        config = json.load(f)


    # Sex,Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight,Class_number_of_rings
    col_to_idx = {col: idx for idx, col in enumerate(df_train.columns)}

    if config['task'] == 'classification':
        cat_indexes = [col_to_idx[col] for col in config['ords']+config['labs']]
    else:
        cat_indexes = [col_to_idx[col] for col in config['ords']]
        
    forest_model = ForestDiffusionModel(np_train, label_y=None, n_t=50, duplicate_K=100, remove_miss=False,
                                        bin_indexes=[], cat_indexes=cat_indexes, int_indexes=[], diffusion_type='flow', n_jobs=-1)
    synth = forest_model.generate(batch_size=n_samples)

    if len(cat_indexes) > 0:
        synthdf = pd.DataFrame(synth, columns=forest_model.X_names_after)
    else:
        synthdf = pd.DataFrame(synth, columns=df_train.columns)
    unassigned = {}
    predummies = list(set(forest_model.X_names_before) - set(forest_model.X_names_after))  
    postdummies = list(set(forest_model.X_names_after) - set(forest_model.X_names_before))  # what their names will be called
    for feat in predummies:
        corresponding = [col for col in postdummies if col.split('_')[0] == feat]
        subdf = synthdf[corresponding] # just the cols for this feature, for all rows
        choices = subdf.to_numpy().argmax(axis=1)
        subdf.iloc[:,:] = 0
        subdf.values[np.arange(len(subdf)), choices] = 1
        synthdf[corresponding] = subdf

    undummies = pd.from_dummies(synthdf[postdummies], sep='_')
    synthdf = pd.concat([synthdf, undummies], axis=1)
    synthdf = synthdf.drop(columns=postdummies)
    synthdf = synthdf[forest_model.X_names_before]
    synthdf.columns = df_train.columns

    synthdf.to_csv(os.path.join(out_path, f'{dataset}_{now}.csv'), index=False)
