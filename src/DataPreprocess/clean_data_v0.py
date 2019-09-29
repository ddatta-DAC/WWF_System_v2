import pandas as pd
import inspect
import numpy as np
import glob
import os
from joblib import Parallel, delayed
import re
import yaml

# global path
cur_path = None


def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

    os.chdir(this_file_path)
    return this_file_path


# ------------------------  #
# Configuration 			#
# ------------------------  #

def get_config():
    CONFIG_FILE = 'data_clean_config.yaml'
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
        return config


def get_hs_code_regex(_dir):
    global cur_path

    file = os.path.join(
        cur_path,
        '..',
        '..',
        'metadata',
        _dir + '_hs_codes.txt'
    )

    lines = [line.rstrip('\n') for line in open(file)]
    _list = []
    for c in lines:
        rpt = str(6 - len(c))
        if len(c) < 6:
            chunk = '(' + c + '.{' + rpt + '})'
        else:
            chunk = '(' + c + ')'
        _list.append(chunk)

    r = '|'.join(_list)
    r = '^' + r + '$'
    return r


# def make_hs_code_2_again(row,attr='HSCode'):
# 	return int(str(row[attr])[:2])
#
# def make_hs_code_4_again(row,attr='HSCode'):
# 	return int(str(row[attr])[:4])
#
# def make_hs_code_6_again(row,attr='HSCode'):
# 	return int(str(row[attr])[:6])

# ------------------------------------ #
def process_file(
        file_path,
        valid_cols,
        op_dir,
        _dir
):
    regex_str = get_hs_code_regex(_dir)

    df = pd.read_csv(
        file_path,
        low_memory=False,
        usecols=valid_cols
    )

    if len(df) == 0:
        return
    df = df.dropna(how='any', subset=['HSCode'])

    df['HSCode'] = df['HSCode'].astype(str)

    def remove_dot(row):
        return row['HSCode'].replace('.', '')

    df['HSCode'] = df.apply(remove_dot, axis=1)

    def filter_hs_code(row, regex_str):
        m = re.match(regex_str, row['HSCode'])
        r = None
        if m is not None:
            r = int(m.group(0))
        return r

    df['HSCode'] = df.apply(
        filter_hs_code,
        axis=1,
        args=(regex_str,)
    )
    df = df.dropna(how='any', subset=['HSCode'])
    df['HSCode'] = df['HSCode'].astype(int)

    df.rename(
        columns={'HSCode': 'hscode_6'},
        inplace=True
    )

    if len(df) == 0:
        return

    f_name = (file_path.split('/')[-1]).split('.')[0] + '_filtered' + '.csv'
    op_file_path = os.path.join(op_dir, f_name)

    df.to_csv(op_file_path, index=None)
    return


'''
Get the attributes 
This is hardcoded in config file
'''


def get_valid_cols(_dir, config):
    valid_cols = config[_dir]['use_cols']
    return valid_cols


'''
Function to process each directory / source
'''


def process_dir(
        data_dir,
        _dir,
        op_path,
        config
):
    n_jobs = 5
    op_dir = os.path.join(op_path, _dir)

    if not os.path.exists(op_dir):
        os.mkdir(op_dir)

    print(' > ', data_dir)
    csv_files_path = os.path.join(
        data_dir,
        _dir
    )

    valid_cols = get_valid_cols(
        _dir,
        config
    )

    files = glob.glob(csv_files_path + '/*.csv')

    Parallel(n_jobs=n_jobs)(
        delayed(process_file)(
            _file, valid_cols, op_dir, _dir
        ) for _file in files)
    return


# ------------------------------------ #
def main():
    config = get_config()
    cur_path = get_cur_path()

    data_dir = os.path.join(
        cur_path,
        '..',
        '..',
        'Data'
    )
    op_path = os.path.join(
        cur_path,
        '..',
        '..',
        'GeneratedData',
        'FilteredData',
    )
    if not os.path.exists(op_path):
        os.mkdir(op_path)

    target_dirs = list(config.keys())

    for _dir in target_dirs:
        if config[_dir]['process']:
            process_dir(
                data_dir,
                _dir,
                op_path,
                config
            )

main()
