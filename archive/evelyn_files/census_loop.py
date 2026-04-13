import lightgbm
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import yaml

sys.path.insert(0, '..')
from preprocess import Preprocess
from modeling_utils import mae_loss, tune_model, rf_train_test_write

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

## change working directory to this script's location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

## load lasso config file
with open('../../config/census_loop_config.yaml', 'r') as stream:
    out = yaml.safe_load(stream)

## load paths, params, and variable lists from config

#paths
data_path = out['paths']['data_path']
fips_path = out['paths']['fips_path']
results_path = out['paths']['results_path']
log_path = out['paths']['log_path']
sampler_path = out['paths']['sampler_path']
params_path = out['paths']['params_path']
trials_path = out['paths']['trials_path']

#params
share_non_null = out['model_params']['share_non_null']
random_state = out['model_params']['random_state']
wins_pctile = out['model_params']['wins_pctile']
log_label = out['model_params']['log_label']
n_trials = out['model_params']['n_trials']
loss_func = out['model_params']['loss_func']

#varbs
continuous_full = out['features']['continuous']
census_full = out['features']['census']
meta = out['features']['meta']
label = out['features']['label']
sale_date_col = out['features']['sale_date']
time_cols = out['features']['time']

## load in full data
print('loading data')
df = pd.read_csv(data_path)
#df = pd.read_csv(data_path, nrows=1000000) # subsample for testing

print('data loaded')

# data cleaning
df = df[df.MULTI_OR_SPLIT_PARCEL_CODE.isnull()]
df = df[~df.fips.isnull()]
#df[sale_date_col] = pd.to_datetime(df[sale_date_col], errors='coerce')
#df[sale_date_col] = df[sale_date_col].dt.strftime("%Y%m%d").astype(int)
df.fips = [int(x) for x in df.fips]

print('data cleaned')

## get list of fips to loop through; determine how many fips have already been processed
fips = pd.read_csv(fips_path)
fips_list = [int(x) for x in fips.fips.unique().tolist()]

if os.path.exists(results_path):
    results = pd.read_csv(results_path)
    completed_fips = [int(x) for x in results.fips]
    remaining_fips = set(fips_list) - set(completed_fips)
    print(f"{len(remaining_fips)} counties remaining out of {len(fips_list)}")
else:
    results = pd.DataFrame(columns=meta+['y_true', 'y_pred', 'ratio', 'model_id'])
    print(f"{len(fips_list)} counties remaining out of {len(fips_list)}")
    remaining_fips = set(fips_list)

random.seed(random_state)
remaining_fips = list(remaining_fips)
random.shuffle(remaining_fips)

for fips in remaining_fips:
    print(f'starting fips {fips}')

    # clear paths
    for temp_path in [sampler_path, params_path, trials_path, 'census_loop.db']:
        if os.path.exists(temp_path):
            os.remove(temp_path)
	
    # subset to sales from fips
    data = df[df.fips == fips].copy()

    # skip fips with fewer than 100 sales in full 5-year sample
    if len(data) < 100:
        continue

    ## designate categorical and continuous
    categorical = []
    continuous = continuous_full
    binary = []

    # skip fips where record of current assessment does not exist for more than 10% of homes  
    if data[continuous[0]].isnull().sum()/data.shape[0] > 0.10:
        continue

    # add census variables to categorical features; subset only to features that are available in data.columns (they should all be there)
    continuous = continuous + census_full
    continuous = [x for x in continuous if x in data.columns]
	
    # preproess
    preproc = Preprocess(data.copy(),
			    label,
			    continuous,
			    binary,
			    categorical,
			    meta,
                sale_date_col,
			    share_non_null=share_non_null,
			    random_state=random_state,
			    wins_pctile=1,
			    log_label=log_label,
                log_dir=log_path
                )

    try:
        X_train, X_test, y_train, y_test, meta_train, meta_test, continuous, binary, categorical = preproc.run(target_encode=False, 
                                                                                                           one_hot=True, 
                                                                                                           drop_lowest_ratios=True,
																										   gen_time_vars=True,
                                                                                                           drop_repeat_sales=True
                                                                                                           )
    except (lightgbm.basic.LightGBMError, ValueError):
        continue
		
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f'train or test set empty after train-test split. Skipping {fips}.')
        continue
	
    # print data info before tuning
    print('X_train, X_test, y_train, y_test info after preprocessing:')
    print(X_train.info())
    print(X_test.info())
    print(y_train.info())
    print(y_test.info())

    ## tune, train, and write output from full-feature model

    # tune
    str_fips = str(fips)
    fips_params_path = re.sub(r'\.pkl', '_' + str_fips + r'.pkl', params_path)

    tune_model(X_train, 
	            y_train,
 	            study_name='census_loop',
                load_if_exists=True,
                sampler=TPESampler(seed=42, n_startup_trials=20, multivariate=True),
                sampler_path=sampler_path, #.pkl file
                params_path=fips_params_path, #.pkl file
                trials_path=trials_path, #.csv file
                geography=None,
                n_trials=n_trials,
                random_state=random_state,
                loss_func=mae_loss,
                subsample_train=False, # to do: add toggle in config
                model='random_forest',
                write_output=True,
                )

    # train
    output = rf_train_test_write(X_train, 
			X_test, 
			y_train, 
			y_test, 
			meta_test,
			params_path=fips_params_path,
            model_id='census_bg_rf',
            log_label=True) # type: ignore
    
    # write
    results = pd.concat([results, output])
    results.to_csv(results_path, index=False)

    # delete temporary paths created during model tuning
    for temp_path in [sampler_path, trials_path, 'census_loop.db']:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"{fips} complete.")