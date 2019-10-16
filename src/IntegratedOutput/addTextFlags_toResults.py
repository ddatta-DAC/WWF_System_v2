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


# ----------------------------------------------------------------------- #
# Append the text flags
# ----------------------------------------------------------------------- #
CONFIG_FILE = './config.yaml'

def set_up_config(
        dir
):
    global CONFIG_FILE
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)
    DIR = dir
    return CONFIG


def aux_1(CONFIG, _tf_file):
    _df = pd.read_csv(
        _tf_file,
        low_memory=False,
        index_col=None
    )

    _df[CONFIG['id_column_name']] = _df[CONFIG['id_column_name']].astype(int)
    _df[CONFIG['text_flag_column_name']] = _df[CONFIG['text_flag_column_name']].astype(int)

    # get the list of IDs that are 1
    col = CONFIG['text_flag_column_name']
    _df = _df.loc[_df[col] == 1]
    _list = list(_df[CONFIG['id_column_name']])
    return _list


def set_flag_aux(df, target_list, flag_column, id_col):
    def check_aux(row, target_list, id_col):
        if row[id_col] in target_list:
            return 1
        return 0

    df[flag_column] = df.apply(check_aux, axis=1, args=(target_list, id_col))
    return df


def append_text_flags(CONFIG, DIR):
    data_loc = os.path.join(CONFIG['TEXT_FLAG_LOC'], DIR)
    tf_file_list = glob.glob(os.path.join(data_loc, 'text_flag**.csv'))
    id_col = CONFIG['id_column_name']

    num_proc = 10
    pool = mp.Pool(processes=num_proc)

    results = [
        pool.apply_async(
            aux_1,
            args=(CONFIG,_tf_file)
        ) for _tf_file in tf_file_list
    ]

    # This list is list of PnjivaRecordIds that have a 1
    target_list = []
    for p in results:
        tmp = p.get()
        target_list.extend(tmp)

    base_file = os.path.join(CONFIG['MODEL_RESULTS_LOC'], DIR, CONFIG['COMBINED_OP_FILE_NAME'])
    base_df = pd.read_csv(base_file, low_memory=False, index_col=None)
    flag_column = CONFIG['text_flag']
    base_df[flag_column] = 0


    # Chunk the data and set the flag
    num_chunks = 100
    list_df = np.array_split(base_df, num_chunks)
    pool = mp.Pool(processes=num_chunks)

    results = [
        pool.apply_async(
            set_flag_aux,
            args=(_df, target_list, flag_column, id_col)
        ) for _df in list_df
    ]
    master_df = None

    for p in results:
        tmp = p.get()
        if master_df is  None:
            master_df = tmp
        else:
            master_df = master_df.append(tmp, ignore_index=True)

    print(master_df)
    op_file = os.path.join(CONFIG['MODEL_RESULTS_LOC'], DIR, CONFIG['COMBINED_SCORES_wTEXT_OP_FILE'])
    master_df.to_csv(
        op_file,
        index=None
    )
    return


# ------ #

def main(dir):
    CONFIG = set_up_config(dir)
    append_text_flags(CONFIG, dir)


# ========================================================================= #

parser = argparse.ArgumentParser(description='Get the final output')
parser.add_argument(
    '--dir',
    nargs='?',
    type=str,
    help=' < Data source > ',
    choices=['us_import', 'china_import', 'china_export', 'peru_export']
)

args = parser.parse_args()

if args.dir is not None:
    print('Calling main :: ', args.dir)
    main(
        dir=args.dir
    )
