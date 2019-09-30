import pandas as pd
import os
import inspect
import yaml
import glob
from joblib import Parallel, delayed

'''
Steps:
1. Extract columns which are used for model
2. Segment the data based data_segment_config.yaml
'''


def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

    os.chdir(this_file_path)
    return this_file_path


def get_config():
    CONFIG_FILE = 'model_data_clean_config.yaml'
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
        return config


def get_valid_cols(_dir, config):
    valid_cols = config[_dir]['use_cols']
    return valid_cols


# -------------------------------------- #

def combine_csv_files_1(file_list, valid_cols):
    df = None
    for f in file_list:
        tmp_df = pd.read_csv(f, usecols=valid_cols, low_memory=False)
        if df is None:
            df = tmp_df
        else:
            df = df.append(tmp_df, ignore_index=True)
    print(' combine_csv_files_1 :> ', len(df))
    return df


'''
Process data for each source
'''


def process_dir(
        data_dir,
        dir,
        op_path,
        config
):

    op_dir = os.path.join(op_path, dir)
    if not os.path.exists(op_dir):
        os.mkdir(op_dir)

    # ------
    # segment_config
    # ------
    SEG_CONFIG_FILE = 'data_segment_config.yaml'
    with open(SEG_CONFIG_FILE) as f:
        segment_config = yaml.safe_load(f)

    inp_files_path = os.path.join(
        data_dir,
        dir
    )

    valid_cols = get_valid_cols(
        dir,
        config
    )
    _seg_conf = segment_config[dir]

    for case, conf in _seg_conf.items():
        train_p = conf['train']
        test_p = conf['test']
        files_test = []
        files_train = []

        for _pattern in test_p:
            f = glob.glob(os.path.join(
                inp_files_path,
                '**' + str(_pattern) + '*.csv')
            )
            files_test.extend(f)
        test_df = combine_csv_files_1(files_test, valid_cols)

        for _pattern in train_p:
            f = glob.glob(os.path.join(
                inp_files_path,
                '**' + str(_pattern) + '*.csv')
            )
            files_train.extend(f)
        train_df = combine_csv_files_1(files_train, valid_cols)

        op_subdir = os.path.join(
            op_dir,
            str(case)
        )
        if not os.path.exists(op_subdir):
            os.mkdir(op_subdir)
        op_f_name_train = 'data_train_case_' + str(case) + '.csv'
        op_f_name_test = 'data_test_case_' + str(case) + '.csv'

        test_df.to_csv(
            os.path.join(
                op_subdir, op_f_name_test
            )
        )

        train_df.to_csv(
            os.path.join(
                op_subdir, op_f_name_train
            )
        )

    return


def main():
    config = get_config()
    cur_path = get_cur_path()

    data_dir = os.path.join(
        cur_path,
        '..',
        '..',
        'GeneratedData',
        'FilteredData'
    )
    op_path = os.path.join(
        cur_path,
        '..',
        '..',
        'GeneratedData',
        'SegmentedData',
    )
    if not os.path.exists(op_path):
        os.mkdir(op_path)

    target_dirs = list(config.keys())

    Parallel(n_jobs=4)(
        delayed(process_dir)(
            data_dir,
            dir,
            op_path,
            config
        ) for dir in target_dirs if config[dir]['process'] )

    return

main()
