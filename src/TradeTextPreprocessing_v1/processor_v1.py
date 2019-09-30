from pprint import pprint
import yaml
import numpy as np
import glob
import inspect
import re
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
nltk.download('stopwords')
nltk.download('omw')
from nltk.corpus import wordnet
import os
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

nlp = spacy.load('en_core_web_md')
from joblib import Parallel, delayed

'''
Set up config
'''
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
    global CONFIG
    config_file = 'text_preproc_config_v1.yaml'
    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)
    return CONFIG

def setup():
    global cur_path
    global CONFIG

    setup_config()
    cur_path = get_cur_path()
    # set up output location
    op_loc = CONFIG['output_loc']

    if not os.path.exists(op_loc):
        os.mkdir(op_loc)
    DIR = CONFIG['DIR']
    if not os.path.exists(os.path.join(op_loc,DIR)):
        os.mkdir(os.path.join(op_loc,DIR))
    return
setup()


'''
Filter out by requisite hs codes - reduces computation
Method is tailored for pandas column vectorized (speed optimal)
'''


def get_file_paths(DATA_DIR):
    print(os.path.join(DATA_DIR, 'panjiva_*.csv'))
    all_files = sorted(glob.glob(
        os.path.join(DATA_DIR, 'panjiva_*.csv')
    ))
    return all_files

def ngrams(input_word_list, n=1):
    output = []
    for i in range(len(input_word_list)-n+1):
        output.append(' '.join(input_word_list[i:i+n]))
    return output


def get_data(DIR, file_path):
    global CONFIG
    usecols = CONFIG[DIR]['usecols']
    print('use cols',usecols)
    print(file_path)
    try:
        df = pd.read_csv(
            file_path,
            usecols=usecols,
            low_memory=False
        )

        df = df.reset_index(drop=True)
        return df
    except:
        print(file_path)
        exit(10)



'''
Function to 
1. clean text
2. check valid word
3. extract unigram + bigram 
'''
def get_external_objects():
    english_vocab = list(set(w.lower() for w in nltk.corpus.words.words()))
    stop_words = list(stopwords.words('english'))
    sysnets = wordnet.synsets
    return english_vocab, stop_words, sysnets

english_vocab, stop_words, sysnets = get_external_objects()
# Call this in a vectorized manner
def extract_kw( desc, MAX_NGRAM_LENGTH):

    global english_vocab
    global stop_words
    global sysnets

    POS_exclusion = [
        'NUM', 'CCONJ', 'CONJ', 'INTJ', 'PART', 'SCONJ', 'SYM', 'X', 'PUNCT'
    ]
    if type(desc)!=str:
        return None
    phrases = desc.split(';')
    '''
    cache_1 : unigrams
    cache_2 : bigrams
    '''
    row_ngrams = []
    for phrase in phrases:
        cleaned_phrase = []
        phrase = phrase.lower()
        doc = nlp(phrase)
        words = word_tokenize(phrase)
        # Per phrase unigrams

        # Per phrase bigrams
        for word in doc:
            w = word.text.lower()
            skip_flag = False
            if len(w) <= 1:
                skip_flag = True
            if w in stop_words:
                skip_flag = True
            if word.pos_ in POS_exclusion:
                skip_flag = True
            if w not in words:
                skip_flag = True
            if skip_flag:
                continue

            in_sysnet = False
            try :
                in_sysnet = sysnets(w)
            except:
                pass

            if in_sysnet and w in english_vocab:
                cleaned_phrase.append(w)

        phrase_ngrams = {}
        for q in range(1, MAX_NGRAM_LENGTH+1):
            ng = ngrams(cleaned_phrase,q)
            phrase_ngrams[q] = ';'.join(ng)
        ng = ';'.join(phrase_ngrams.values())
        row_ngrams.append(ng)

    row_ngrams = ';'.join(row_ngrams)
    return row_ngrams


def divorce_ngrams(_string, sep='*;*', n=1):
    res = _string.split(sep)
    return str(res[n - 1])


# ============ #
# Divide the main data-frame into chunks
# ============ #
def divide_DF(inp_df, num_chunks=10):
    list_dfs = np.array_split(inp_df, num_chunks)
    return list_dfs

# =============== #
# match keywords based on a keywords list
# keywords list should be curated from the different sources
# =============== #
def match_keywords(row , keywords_list):
    if type(row['ngrams'])!= str : return 0
    candidates = (row['ngrams']).split(';')
    for c in candidates:
        if c in keywords_list:
            return 1
    return 0

# ============ #
# Get the uni-grams and bi-grams
# ============ #
def process(
        df,
        DIR,
        master_keywords_list
):
    text_col = CONFIG[DIR]['text_col']
    MAX_NGRAM_LENGTH = CONFIG['MAX_NGRAM_LENGTH']
    df = df.reset_index(drop=True)
    desc_series = df[text_col]

    res = desc_series.apply(
        extract_kw,
        args=(MAX_NGRAM_LENGTH,)
    )
    df['ngrams'] = res

    # Now search for matches
    df['kw_flag'] = df['ngrams'].apply(
        match_keywords,
        args = (master_keywords_list,)
    )
    # df = df.loc[df['kw_flag']==1]

    # try:
    #     del df['ngrams']
    #     del df['kw_flag']
    # except:
    #     pass

    return df

def get_master_keywords_list():
    global CONFIG
    res = []
    loc = CONFIG['source_keywords_loc']
    files = CONFIG['source_keywords_files']
    for f in files:
        f_path = os.path.join(loc, f)
        _tmp = pd.read_csv(f_path,header=None, index_col=None)
        wordList = list(_tmp[0])
        wordList = [_.lower() for _ in wordList]
        res.extend(wordList)
    res = list(set(res))
    return res

def nlp_task(df, DIR, num_jobs=100):

    global CONFIG
    master_keywords_list = get_master_keywords_list()
    hscode_col = CONFIG[DIR]['hscode_col']
    text_col = CONFIG[DIR]['text_col']

    list_dfs = divide_DF(df)
    result_df_list = Parallel(
        n_jobs=num_jobs,
        prefer="threads")(
        delayed(process)
        (_df, DIR , master_keywords_list) for _df in list_dfs
    )

    result_df = None
    # -----
    # Join the chunks
    # -----
    for _df in result_df_list:
        cols_to_remove = [hscode_col, text_col]
        for c in cols_to_remove:
            try:
                del _df[c]
            except:
                pass
        if result_df is None:
            result_df = pd.DataFrame(_df, copy=True)
        else:
            result_df = result_df.append(_df, ignore_index=True)

    return result_df

def main():
    global CONFIG

    DATA_DIR = os.path.join(
        CONFIG['DATA_DIR'],
        CONFIG['DIR']
    )
    DIR = CONFIG['DIR']

    # List of all the files
    files_paths = get_file_paths(DATA_DIR)

    op_loc = CONFIG['output_loc']
    op_df_loc = os.path.join(
        op_loc,
        DIR
    )

    for file_path in files_paths:
        df = get_data(DIR, file_path)
        df = nlp_task(
            df,
            DIR,
            100
        )
        _tmp = '_'.join((file_path.split('/'))[-1].split('_')[-2:])
        df_name = 'text_flags_' + _tmp
        print(df_name)
        _df_path = os.path.join(op_df_loc, df_name)
        df.to_csv(_df_path, index=False)
    return

main()