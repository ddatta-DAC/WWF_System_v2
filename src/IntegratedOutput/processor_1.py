import glob
import pandas as pd
import numpy as np
import multiprocessing as mp
import os
import inspect
import yaml
import sys
import argparse
import math
from sklearn import preprocessing
CONFIG_FILE = './config.yaml'
# ----

def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

    os.chdir(this_file_path)
    return this_file_path


def set_up_config(
        dir
):
    global CONFIG_FILE
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    DIR = dir
    MODEL_RESULTS_DIR =  CONFIG['MODEL_RESULTS_LOC']
    SUBDIRS = []
    for f in sorted( glob.glob( MODEL_RESULTS_DIR + '/**')) :
        SUBDIRS.append( f.split('/')[-1])

    return CONFIG ,MODEL_RESULTS_DIR, SUBDIRS


# Apply HS Code based filtering

'''
Collate the scored records from model output
'''
def collate_scored_records (CONFIG, DATA_DIR , DIR):

    files = sorted(glob.glob(DATA_DIR+ '/' + DIR + '/**/**.csv'))
    print(files)
    master_df = None
    for f in files :
        df = pd.read_csv(
            f, index_col=None , low_memory=False
        )
        # Scale the results between 0 and 1 for each
        x = np.reshape(df['score'].values,[-1,1])
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['score'] = np.reshape(x_scaled,[-1])

        if master_df is not None:
            master_df =  master_df.append(df,ignore_index=True)
        else:
            master_df = df

    # Output the file
    op_f_path = os.path.join(
        DATA_DIR,
        DIR,
        CONFIG['COLLATED_MODEL_OP_FILE_NAME']
    )
    master_df.to_csv(op_f_path)
    return master_df

def get_HDF_df(CONFIG, DIR):
    _path = os.path.join( CONFIG['HDF_OUTPUT_LOC'], DIR)
    f_name = 'HDF_results.csv'
    f_path = os.path.join(_path,f_name)
    df = pd.read_csv(f_path,index_col=None, low_memory=False)
    return df

def combine(model_op_df, hdf_op_df):
    id_col = 'PanjivaRecordID'
    valid_ids = list(model_op_df[id_col])
    hdf_df = pd.DataFrame(
        hdf_op_df.loc[hdf_op_df[id_col].isin(valid_ids)]
    )
    combined_df = model_op_df.merge(hdf_df, how='inner', on =id_col)

    # ==========
    # Set up combined hdf score first
    # ==========

    def get_combined_hdf_score(row):
        _list = ['LEB_flag', 'CITES_flag', 'WWF_HighRisk_flag', 'IUCN_RedList_flag']
        _sum = 0
        for c in _list:
            _sum += row[c]
        return _sum

    def calculate_combined_score(row,_max):
        s1 = 1 - row['score']
        s2 = row['hdf_score']
        s1* math.exp(s2/_max)

    combined_df['hdf_score'] = 0
    combined_df['hdf_score'] = combined_df.apply(
        get_combined_hdf_score, axis = 1
    )

    combined_df['combined_score'] = 0
    _max = max(list(combined_df['hdf_score']))
    combined_df['combined_score'] = combined_df.apply(
        calculate_combined_score, axis= 1, args=(_max,)
    )
    return  combined_df

def main(dir):
    DIR = dir
    CONFIG , MODEL_RESULTS_DIR, SUBDIRS = set_up_config(DIR)
    model_op_df =  collate_scored_records (CONFIG, MODEL_RESULTS_DIR, DIR)
    hdf_op_df = get_HDF_df(CONFIG, DIR)
    # ================
    # Combine the model output and HDF output
    # ================

    final_df = combine(model_op_df, hdf_op_df)
    OP_LOC = CONFIG['MODEL_RESULTS_LOC']
    f_name = CONFIG['COMBINED_OP_FILE_NAME']
    f_path = os.path.join(OP_LOC,DIR, f_name)
    final_df.to_csv(
        f_path,
        index=None
    )


parser = argparse.ArgumentParser(description='Get the final output')
parser.add_argument(
    '--dir',
    nargs='?',
    type=str,
    help=' < Data source > ',
    choices=['us_import', 'china_import', 'china_export','peru_export']
)

args = parser.parse_args()

if args.dir is not None :
    print('Calling main :: ', args.dir)
    main(
        dir = args.dir
    )