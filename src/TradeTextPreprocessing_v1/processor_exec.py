import multiprocessing as mp
import os
import inspect
import yaml
import sys
import glob



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

    config_file = 'text_preproc_config_v1.yaml'
    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)
    return CONFIG

def setup():

    CONFIG = setup_config()
    cur_path = get_cur_path()
    # set up output location
    op_loc = CONFIG['output_loc']

    if not os.path.exists(op_loc):
        os.mkdir(op_loc)
    DIR = CONFIG['DIR']
    if not os.path.exists(os.path.join(op_loc,DIR)):
        os.mkdir(os.path.join(op_loc,DIR))
    return CONFIG, cur_path

def get_file_paths(DATA_DIR):
    print(os.path.join(DATA_DIR, 'panjiva_*.csv'))
    all_files = sorted(glob.glob(
        os.path.join(DATA_DIR, 'panjiva_*.csv')
    ))
    return all_files


def process_data(CONFIG,file_path):
    try:
        from . import processor_v1
    except:
        import processor_v1

    r = processor_v1.invoke(CONFIG, file_path)
    return r

def main():
    CONFIG, cur_path= setup()

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
    results = [pool.apply_async( process_data, args=(CONFIG, file_path,)) for file_path in files_paths[:4] ]
    output = [p.get() for p in results]
    print(output)
    return

main()
