import pandas as pd
import os
import sys
import glob
import yaml
import inspect
import argparse
import pickle
sys.path.append('./../..')
sys.path.append('./..')

try:
    from . import  tf_model as tf_model
except:
    import tf_model as tf_model
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


def set_up_config(
        dir,
        case
):
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
    DATA_DIR = os.path.join(
        CONFIG['DATA_DIR'],
        DIR,
        SUB_DIR
    )

    if not os.path.exists(os.path.join(CONFIG['model_save_dir'])):
        os.mkdir(os.path.join(CONFIG['model_save_dir']))
    if not os.path.exists(os.path.join(CONFIG['model_save_dir'],DIR)):
        os.mkdir(os.path.join(CONFIG['model_save_dir'],DIR))
    if not os.path.exists(os.path.join(CONFIG['model_save_dir'], DIR, SUB_DIR)):
        os.mkdir(os.path.join(CONFIG['model_save_dir'], DIR, SUB_DIR))

    MODEL_SAVE_DIR = os.path.join(
        CONFIG['model_save_dir'],
        DIR,
        SUB_DIR
    )
    print(' ... ', MODEL_SAVE_DIR)
    if not os.path.exists(CONFIG['OP_DIR']):
        os.mkdir(os.path.join(CONFIG['OP_DIR']))
    if not os.path.exists(os.path.join(CONFIG['OP_DIR'], DIR)):
        os.mkdir(os.path.join(CONFIG['OP_DIR'], DIR))
    # if not os.path.exists(os.path.join(CONFIG['OP_DIR'], DIR, SUB_DIR)):
    #     os.mkdir(os.path.join(CONFIG['OP_DIR'], DIR, SUB_DIR))

    OP_DIR = os.path.join(
        CONFIG['OP_DIR'],
        DIR,
        SUB_DIR
    )

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    if not os.path.exists(CONFIG['RESULT_DIR']):
        os.mkdir(os.path.join(CONFIG['RESULT_DIR']))
    if not os.path.exists(os.path.join(CONFIG['RESULT_DIR'], DIR)):
        os.mkdir(os.path.join(CONFIG['RESULT_DIR'], DIR))

    RESULT_DIR = os.path.join(
        CONFIG['RESULT_DIR'],
        DIR,
        SUB_DIR
    )
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    return CONFIG, DATA_DIR, MODEL_SAVE_DIR, OP_DIR, RESULT_DIR, SUB_DIR


def get_domain_dims(DATA_DIR):
    f_path = os.path.join(DATA_DIR, 'domain_dims.pkl')
    with open(f_path, 'rb') as fh:
        res = pickle.load(fh)
    return list(res.values())


def fetch_data(DATA_DIR):
    id_col = 'PanjivaRecordID'
    domain_dims = get_domain_dims(DATA_DIR)

    train_x_pos_file = os.path.join(
        DATA_DIR,
        'matrix_train_positive_v1.pkl'
    )

    with open(train_x_pos_file, 'rb') as fh:
        train_x_pos = pickle.load(fh)

    train_x_neg_file = os.path.join(
        DATA_DIR,
        'negative_samples_v1.pkl'
    )

    with open(train_x_neg_file, 'rb') as fh:
        train_x_neg = pickle.load(fh)
        train_x_neg = train_x_neg

    test_x_file = os.path.join(
        DATA_DIR,
        'test_data.csv'
    )
    test_df = pd.read_csv(test_x_file,index_col=None)
    test_normal_idList = list(test_df[id_col])

    del test_df[id_col]
    test_x = test_df.values
    test_pos = [test_normal_idList, test_x]

    return train_x_pos, train_x_neg, test_pos, domain_dims


def set_up_model(
        config,
        DATA_DIR,
        SAVE_DIR,
        OP_DIR,
        dir,
        case
):
    SUBDIR =  str(case)

    embedding_dims = config[dir]['emb_dims']
    MODEL_NAME = '_'.join( [config['MODEL_NAME'], 'case', SUBDIR])
    model_obj = tf_model.model(
        MODEL_NAME,
        SAVE_DIR,
        OP_DIR
    )

    model_obj.set_model_options(
        show_loss_figure=config[dir]['show_loss_figure'],
        save_loss_figure=config[dir]['save_loss_figure']
    )

    domain_dims = get_domain_dims(DATA_DIR)
    LR = config[dir]['learning_rate']
    model_obj.set_model_hyperparams(
        domain_dims=domain_dims,
        emb_dims=embedding_dims,
        batch_size=config[dir]['batchsize'],
        num_epochs=config[dir]['num_epochs'],
        learning_rate=LR,
        num_neg_samples=config[dir]['num_neg_samples']
    )

    model_obj.inference = False
    model_obj.build_model()
    return model_obj


def main( dir, case ):
    CONFIG, DATA_DIR, MODEL_SAVE_DIR, OP_DIR, RESULT_DIR, SUB_DIR = set_up_config( dir, case )
    if CONFIG[dir]['process'] is False:
        return 0

    model_obj = set_up_model(
        CONFIG,
        DATA_DIR,
        MODEL_SAVE_DIR,
        OP_DIR,
        dir,
        case
    )

    train_x_pos, train_x_neg, test_pos, domain_dims  = fetch_data(DATA_DIR)

    model_obj.train_model(
        train_x_pos,
        train_x_neg
    )
    test_id_list = test_pos[0]
    test_x_data =  test_pos[1]

    scores = model_obj.get_event_score(test_x_data)

    columns = [CONFIG['id_col'],'score']
    result_df = pd.DataFrame(columns = columns )
    result_df[columns[0]] =  test_id_list
    result_df[columns[1]] =  scores
    result_df = result_df.sort_values(
        by=['score']
    )
    results_op_path = os.path.join(
        RESULT_DIR,
        'recordID_scores.csv'
    )
    result_df.to_csv(
        results_op_path,
        index=None
    )
    return 1


# ------------------------------------------------------------------------------- #
print(' ------ ')
parser = argparse.ArgumentParser(description='Train the models')
parser.add_argument(
    '--dir',
    nargs='?',
    type=str,
    help=' < Data source > ',
    choices=['us_import', 'china_import', 'china_export','peru_export']
)
parser.add_argument(
    '--case',
    nargs='?',
    type=int,
    help=' < Segmented Case > '
)


args = parser.parse_args()

if args.dir is not None and args.case is not None:
    print('Calling main :: ', args.dir, args.case)
    main(
        dir = args.dir,
        case = args.case
    )


