import multiprocessing as mp
import os
import inspect
import yaml
import sys
import glob
import argparse

try:
    from src.ml_model import main_model_exec
except:
    import main_model_exec

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

def setup():

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

    return exec_dict





def process_data(CONFIG, file_path):

    try:
        from . import processor_v1
    except:
        import processor_v1

    r = processor_v1.invoke(CONFIG, file_path)
    return r

def main():

    # List of all the files
    exec_dict = setup()

    num_proc = 5
    pool = mp.Pool(processes=num_proc)
    print(pool)
    for _dir,cases in exec_dict.items():
        results = [pool.apply_async( main_model_exec.main, args=(_dir, case,)) for case in cases ]
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

main()