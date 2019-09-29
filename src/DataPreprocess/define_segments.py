import glob
import pandas as pd
import os
import sys
import yaml
import inspect


def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

    os.chdir(this_file_path)
    print(os.getcwd())
    return this_file_path


def create_segments(
        DIR,
        DATA_LOC,
        train_window=4,
        test_window=2
):

    min_mm = 1
    min_yy = 2015
    max_mm = 12
    max_yy = 2017

    signature_dict = {}

    _mm = min_mm
    _yy = min_yy
    _idx = 1
    while _mm <= max_mm and _yy <= max_yy:
        signature_dict[_idx] = str(_mm).zfill(2) + '_' + str(_yy)
        _idx += 1
        _mm += 1

        if _mm > 12:
            _mm = 1
            _yy += 1

    if (len(signature_dict) - train_window) % test_window != 0:
        print(' Check window sizes for segmentation!!')
        raise ValueError

    num_cases = int((len(signature_dict) - train_window) / test_window)

    segment_dict = {}
    for i in range(1, num_cases + 1):
        st_tr_idx = (i - 1) * 3 + 1
        end_tr_idx = st_tr_idx + train_window - 1
        st_t_idx = st_tr_idx + train_window
        end_t_idx = st_t_idx + test_window - 1

        _train_files = []
        for j in range(st_tr_idx, end_tr_idx + 1):
            sig = signature_dict[j]
            _train_files.append(sig)

        _test_files = []
        for j in range(st_t_idx, end_t_idx + 1):
            sig = signature_dict[j]
            _test_files.append(str(sig))
        segment_dict[i] = {
            'train': _train_files,
            'test': _test_files
        }

    return segment_dict


# ------------------------------------ #
'''
Window lengths:

'us_import'  : train = 6, test = 3
'china_export' : train = 6  test = 3
'china_import' : train = 9 test = 3
'peru_export' : train = 24 test = 12
'''


def main():
    old_path = os.getcwd()
    cur_path = get_cur_path()
    os.chdir(cur_path)

    DATA_LOC = './../../Data'
    config_dict = {
        'us_import': {'train': 6,
                      'test': 3 },
        'china_export': {'train': 6,
                         'test': 3},
        'china_import': {'train': 8,
                         'test': 4},
        'peru_export': {'train': 24,
                        'test': 6}
    }

    result = {}
    for key, conf in config_dict.items():
        _res = create_segments(
            key,
            DATA_LOC,
            conf['train'],
            conf['test']
        )

        result[key] = _res

    # write to yaml
    with open('data_segment_config.yaml', 'w') as outfile:
        yaml.dump(
            result,
            outfile
        )

    os.chdir(old_path)
    return


main()
