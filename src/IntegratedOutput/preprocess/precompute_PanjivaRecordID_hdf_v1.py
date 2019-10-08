import pandas as pd
import os
import sys
import yaml
import glob
sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../..')

try:
    from .src.utils import country_iso_fetcher
except:
    from src.utils import country_iso_fetcher

# Separate out the PanjivaRecordID s
# that correspond to 1 , 2 or 3 human defined filters


def process_files(
        file_paths,
        save_dir,
        metadata_df_1,
        metadata_2_dict,
        country_col
):
    id_list_1 = []
    id_list_2 = []
    id_list_3 = []

    'hscode_6'
    list_1 = list(metadata_df_1[metadata_df_1['lacey'] == 1]['hscode_6'])
    list_2 = list(metadata_df_1[metadata_df_1['plant'] == 1]['hscode_6'])


    for file in file_paths:
        _cols_ = ['PanjivaRecordID','hscode_6']

        if country_col is not None:
            _cols_.append(country_col)
        print(_cols_)
        _df = pd.read_csv(file,usecols=_cols_,low_memory=False)

        # convert country to iso code
        if country_col is not None:
            def get_iso(row):
                res = row[country_col]
                res = country_iso_fetcher.ISO_CODE_OBJ.get_iso_code(res)
                return res

            _df[country_col] = _df.apply(get_iso,axis=1)

        res_1 = list(_df.loc[_df['hscode_6'].isin(list_1)]['PanjivaRecordID'])
        res_2 = list(_df.loc[_df['hscode_6'].isin(list_2)]['PanjivaRecordID'])

        def aux_leb(row):
            c = row[country_col]
            h = int(row['hscode_6'])
            if h in metadata_2_dict.keys() and c in metadata_2_dict[h]:
                return 1
            else:
                return 0

        id_list_1.extend(res_1)
        id_list_2.extend(res_2)

        if country_col is not None:
            _df['leb'] = 0
            _df['leb'] = _df.apply(aux_leb,axis=1)
            res_3 = list(_df.loc[_df['leb']==1]['PanjivaRecordID'])
            id_list_3.extend(res_3)




    # Save the data  files
    op_file = os.path.join(save_dir,'Panjiva_records_hdf_1.txt')
    with open(op_file, 'w') as f:
        for item in id_list_1:
            f.write("%s\n" % item)

    op_file = os.path.join(save_dir, 'Panjiva_records_hdf_2.txt')
    with open(op_file, 'w') as f:
        for item in id_list_2:
            f.write("%s\n" % item)

    if country_col is not None:
        op_file = os.path.join(save_dir, 'Panjiva_records_hdf_3.txt')
        with open(op_file, 'w') as f:
            for item in id_list_3:
                f.write("%s\n" % item)

    return

def main():
    CONFIG = None
    CONFIG_FILE = 'precompute_PanjivaRecordID_hdf.yaml'
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)


    SAVE_DIR = CONFIG['TARGET_DIR']
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    dirs = CONFIG['dirs']
    metadata_df_1 = pd.read_csv(CONFIG['hs_code_metadata_file'])
    metadata_df_2 = pd.read_csv(CONFIG['leb_metadata_file'])

    metadata_2_dict = {}
    # split metadata_df_2 into a dictionary
    for _,row in metadata_df_2.iterrows():
        k = int(row['hscode_6'])
        v = [_ for _ in row['CountryOfOrigin'].split(';')]
        metadata_2_dict[k] = v

    for _dir in dirs:
        data_dir = os.path.join(CONFIG['SRC_DIR'],_dir)
        all_files = glob.glob(
            os.path.join(
                data_dir,
                '*filtered.csv'
            )
        )

        save_dir = os.path.join(
            CONFIG['TARGET_DIR'],
            _dir
        )

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print('Processing ',_dir)
        country_col = CONFIG[_dir]['CountryOfOrigin']
        if country_col == 'None':
            country_col = None

        process_files(all_files, save_dir,metadata_df_1,metadata_2_dict,country_col)

main()







