import multiprocessing as mp
import os
import inspect
import yaml
import sys
import glob
import argparse


CONFIG = None
cur_path =  None
def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )
    os.chdir(this_file_path)
    return this_file_path

def setup_config():
    config_file = 'model_config.yaml'
    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)
    return CONFIG

def setup(dir=None):

    CONFIG = setup_config()

    # Find the 4 datasets
    data_dir = CONFIG['DATA_DIR']
    directory_list = sorted(
        glob.glob(os.path.join(data_dir,'**'))
    )
    print(directory_list)
    exec_dict = {}
    # Get DIR name
    for _ in directory_list:
        dir = _.split('/')[-1]
        print(dir)
        subdirs =  sorted(
            glob.glob(os.path.join(_,'**'))
        )
        cases = [ subdir.split('/')[-1] for subdir in subdirs ]
        print(cases)
        exec_dict[dir] =  cases


    exit(1)
    if dir is not None:

        CONFIG['DIR'] = dir
        DIR = dir
    else:
        DIR = CONFIG['DIR']

    cur_path = get_cur_path()
    # set up output location
    op_loc = CONFIG['output_loc']

    if not os.path.exists(op_loc):
        os.mkdir(op_loc)

    if not os.path.exists(os.path.join(op_loc,DIR)):
        os.mkdir(os.path.join(op_loc,DIR))
    return CONFIG, cur_path

def get_file_paths(DATA_DIR):
    print(os.path.join(DATA_DIR, 'panjiva_*.csv'))
    all_files = sorted(glob.glob(
        os.path.join(DATA_DIR, 'panjiva_*.csv')
    ))
    return all_files


def process_data(CONFIG, file_path):

    try:
        from . import processor_v1
    except:
        import processor_v1

    r = processor_v1.invoke(CONFIG, file_path)
    return r

def main(dir):

    CONFIG, cur_path= setup(dir)
    DATA_DIR = os.path.join(
        CONFIG['DATA_DIR'],
        CONFIG['DIR']
    )

    # List of all the files
    files_paths = get_file_paths(DATA_DIR)
    num_files = len(files_paths)
    num_proc = min(40,num_files)
    pool = mp.Pool(processes=num_proc)
    print(pool)
    results = [pool.apply_async( process_data, args=(CONFIG, file_path,)) for file_path in files_paths ]
    output = [p.get() for p in results]
    print(output)
    return

# # ------------------------------------------------------------------------------- #
# parser = argparse.ArgumentParser(description='Generate data for the ML model')
# parser.add_argument(
#     '--dir',
#     nargs='?',
#     type=str,
#     help=' < Data source > ',
#     choices=['us_import', 'china_import', 'china_export','peru_export']
# )
#
#
# args = parser.parse_args()
# print('Calling data_generator:::', args.dir, args.case)
# main(
#     dir = args.dir
# )

setup()
