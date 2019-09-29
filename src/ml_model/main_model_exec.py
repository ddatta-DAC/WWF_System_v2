import pandas as pd
import os
import sys
import glob
import yaml
import inspect

# ----
CONFIG_FILE = './model_config.yaml'
# ----
def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

    os.chdir(this_file_path)
    return this_file_path


def set_up_config(dir,case):
    global CONFIG_FILE
    global use_cols
    global freq_bound
    global DIR
    global save_dir
    global column_value_filters
    global num_neg_samples
    global SUB_DIR
    global INPUT_DIR

    SUB_DIR = str(case)
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    DIR = dir
    INPUT_DIR = os.path.join(
        CONFIG['inp_dir'],
        DIR,
        SUB_DIR
    )
    if not os.path.exists(INPUT_DIR):
        return False
    if not os.path.exists(os.path.join(CONFIG['save_dir'])):
        os.mkdir(os.path.join(CONFIG['save_dir']))
    if not os.path.exists(os.path.join(CONFIG['save_dir'],DIR)):
        os.mkdir(os.path.join(CONFIG['save_dir'],DIR))

    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR,
        SUB_DIR
    )

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = CONFIG[DIR]['use_cols']
    freq_bound = CONFIG[DIR]['low_freq_bound']
    freq_bound = CONFIG[DIR]['low_freq_bound']
    column_value_filters = CONFIG[DIR]['column_value_filters']
    num_neg_samples = CONFIG[DIR]['num_neg_samples']
    return True


