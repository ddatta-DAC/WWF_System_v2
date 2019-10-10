import pandas as pd
import os
import sys
import yaml
import glob
import time
import multiprocessing as mp
sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../..')
try:
    from src.IntegratedOutput.preprocess.country_iso_fetcher import ISO_CODE_OBJ
except:
    from .country_iso_fetcher import ISO_CODE_OBJ


'''
Perform  LEB based checks
LEB data 2 columns : hscode_6, CountryOfOrigin 
'''


def write_df_WD(CONFIG, DIR, f_name, df):
    working_dir = os.path.join(CONFIG['Working_Dir'],DIR)

    f_path = os.path.join(working_dir, f_name)
    df.to_csv(f_path, index=None)


def read_df_WD(CONFIG, DIR, f_name):
    working_dir = os.path.join(CONFIG['Working_Dir'],DIR)
    f_path = os.path.join(working_dir, f_name)
    df = pd.read_csv(f_path, index_col=None, low_memory=False)
    return df


def LEB_check_aux(row, LEB_df, target_col):
    hscode_col = 'hscode_6'

    hsc = row[hscode_col]

    if hsc not in list(LEB_df[hscode_col]):
        return 0
    else:
        try:
            idx = LEB_df.loc[LEB_df[hscode_col] == hsc].index.tolist()[0]
            _list_countries = (LEB_df.at[idx, 'CountryOfOrigin']).split(';')
            if row[target_col] in _list_countries:
                return 1
        except:
            return 0
    return 0


def LEB_file_proc(file_path, CONFIG, DIR, LEB_df):
    df = pd.read_csv(
        file_path,
        low_memory=False,
        usecols=CONFIG[DIR]['LEB_columns'],
        index_col=None
    )

    # Convert to iso code

    target_col = CONFIG[DIR]['CountryOfOrigin']
    if target_col is False:
        df['LEB_flag'] = 0

    else:

        df[target_col] = df[target_col].apply(ISO_CODE_OBJ.get_iso_code)
        df['LEB_flag'] = 0
        df['hscode_6'] = df['hscode_6'].astype(str)
        df['LEB_flag'] = df.apply(LEB_check_aux, axis=1, args=(LEB_df, target_col))
        del df[target_col]
    # ======
    # Write df to processing temp location
    # ======
    f_name = 'tmp_' + file_path.split('_')[-1]


    write_df_WD(CONFIG, DIR, f_name, df)
    return True


def get_LEB_match_records(CONFIG, DIR):
    if CONFIG[DIR]['CountryOfOrigin'] is False:
        return None

    LEB_df = pd.read_csv(CONFIG['LEB_DATA_FILE'], low_memory=False, index_col=None)
    LEB_df['hscode_6'] = LEB_df['hscode_6'].astype(str)

    # ============
    # These are the segmented files , with actual data though cleaned through initial processing
    # ============
    file_list = sorted(glob.glob(
        os.path.join(
            CONFIG['Data_RealSegmented_LOC'], DIR, '**', 'data_test_**.csv')
    ))

    import multiprocessing as mp
    num_proc = 10
    pool = mp.Pool(processes=num_proc)
    print(pool)

    results = [
        pool.apply_async(
            LEB_file_proc,
            args=(file_path, CONFIG, DIR, LEB_df,)
        ) for file_path in file_list
    ]
    output = [p.get() for p in results]
    print (output)

    return None

# ===========================
#  CITES check
# ===========================
def HSCode_check_aux(row, hscode_list):
    hscode_col = 'hscode_6'
    hsc = row[hscode_col]
    if hsc  in hscode_list:
        return 1
    else:
        return 0


def FLAG_file_proc(file_path, CONFIG, DIR, hscode_list, flag_column):

    df = pd.read_csv(
        file_path,
        low_memory=False,
        index_col=None
    )

    df[flag_column] = 0
    df['hscode_6'] = df['hscode_6'].astype(str)
    df[flag_column] = df.apply(HSCode_check_aux, axis=1, args=(hscode_list,))

    # ======
    # Write df to processing temp location
    # ======
    f_name = 'tmp_' + file_path.split('_')[-1]
    write_df_WD(CONFIG, DIR, f_name, df)
    return True


def common_dispatcher(CONFIG, DIR, hscode_list, flag_column):
    # ============
    # The input now is from Working_Dir
    # ============
    file_list = sorted(glob.glob(
        os.path.join(
            CONFIG['Working_Dir'], DIR, '**.csv')
    ))

    num_proc = 10
    pool = mp.Pool(processes=num_proc)

    results = [
        pool.apply_async(
            FLAG_file_proc,
            args=(file_path, CONFIG, DIR, hscode_list,flag_column, )
        ) for file_path in file_list
    ]
    output = [p.get() for p in results]
    print(output)
    return True


def get_match_records(CONFIG, DIR):

    sources = [ 'CITES', 'WWF_HighRisk', 'IUCN_RedList']

    for source in sources:
        flag_column = source + '_flag'
        data_file_key = source + '_DATA_FILE'

        source_df = pd.read_csv(
            CONFIG[data_file_key],
            low_memory=False,
            index_col=None,
            header=None
        )

        hscode_list = list(source_df[0])
        t1 = time.time()
        common_dispatcher(CONFIG, DIR, hscode_list, flag_column)
        t2 = time.time()
        print(' Time for ' + source + ' checks ', t2 - t1)




def main():
    CONFIG_FILE = 'precompute_PanjivaRecordID_hdf.yaml'
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if not os.path.exists(CONFIG['Working_Dir']):
        os.mkdir(CONFIG['Working_Dir'])
    if not os.path.exists(CONFIG['HDF_OUTPUT_LOC']):
        os.mkdir(CONFIG['HDF_OUTPUT_LOC'])

    process_DIRS = CONFIG['process_dirs']

    for DIR in process_DIRS:

        if not os.path.exists(
                os.path.join(CONFIG['Working_Dir'], DIR)
        ):
            os.mkdir(os.path.join(CONFIG['Working_Dir'], DIR))

        if CONFIG[DIR]['process_LEB']:
            t1 = time.time()
            get_LEB_match_records(CONFIG, DIR)
            t2 = time.time()
            print(' Time for LEB checks ', t2 - t1)



        get_match_records(CONFIG, DIR)

        # =====
        # Combine the files
        # =====
        file_loc = os.path.join(CONFIG['Working_Dir'], DIR)
        file_list = sorted(glob.glob(
            os.path.join(file_loc,'**.csv'))
        )

        master_df = None
        for _file in file_list:
            _tmpdf = pd.read_csv(_file, index_col=None,low_memory=False)
            if master_df is None:
                master_df = _tmpdf
            else:
                master_df = master_df.append(_tmpdf, ignore_index=True)
        op_loc = os.path.join(CONFIG['HDF_OUTPUT_LOC'],DIR)
        if not os.path.exists(op_loc):
            os.mkdir(op_loc)

        f_name = 'HDF_results.csv'
        op_f_path = os.path.join(
            op_loc, f_name
        )
        master_df.to_csv(op_f_path,index=None)
    return


main()
