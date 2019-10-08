import glob
import pandas as pd
import numpy as np
import multiprocessing as mp
import os
import inspect
import yaml
import sys
import argparse

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
    MODEL_RESULTS_DIR = os.path.join(
        CONFIG['MODEL_RESULTS_LOC'],
        DIR
    )
    SUBDIRS = []
    for f in sorted( glob.glob( MODEL_RESULTS_DIR + '/**')) :
        SUBDIRS.append( f.split('/')[-1])

    return MODEL_RESULTS_DIR, SUBDIRS


# Apply HS Code based filtering

'''
Collate the scored records from model output
'''
def collate_scored_records (DATA_DIR):
    files = sorted(glob.glob(DATA_DIR+'/**/**.csv'))
    print(files)


MODEL_RESULTS_DIR, SUBDIRS = set_up_config('china_export')
collate_scored_records (MODEL_RESULTS_DIR)
