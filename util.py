"""
Utilities for downloading data from WMT, tokenizing, vocabularies.

Original taken from
https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/models/rnn/translate/data_utils.py
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import os
import tarfile
from codecs import open
import cPickle as pickle
import shutil


from tensorflow.models.rnn.translate import data_utils

# Tokenizer
script_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scripts')
ja_tokenizer = os.path.join(script_dir, 'ja_tokenizer.py')
en_tokenizer = os.path.join(script_dir,  'tokenizer.pl')

def dump_to_file(filename, obj):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, file=outfile)
    return

def load_from_dump(filename):
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    return obj

def get_tanaka_corpus(directory):
    url = 'ftp://ftp.monash.edu.au/pub/nihongo/examples.utf.gz'
    filename = 'examples.utf.gz'
    if not (os.path.exists(os.path.join(directory, 'tanaka.ja')) and
            os.path.exists(os.path.join(directory, 'tanaka.en'))):
        gz_file = data_utils.maybe_download(directory, filename, url)
        data_file = os.path.join(directory, filename[:-3])
        data_utils.gunzip_file(gz_file, data_file)
        print("Extracting file %s" % data_file)
        with open(data_file, "r") as corpus:
            c = []
            idx = []
            for line in corpus.readlines():
                if line.startswith('A:'):
                    l = line.split('\t')
                    if len(l) == 2:
                        m = l[1].split('#ID=')
                        dic = {
                            'ja': l[0].strip('A: '),
                            'en': m[0].strip()
                        }
                        c.append(dic)
                        idx.append(m[1].strip())

            df = pd.DataFrame(c, index=idx)
            df['en'].to_csv(os.path.join(directory, 'tanaka.en'), sep='\t', header=False, index=False)
            df['ja'].to_csv(os.path.join(directory, 'tanaka.ja'), sep='\t', header=False, index=False)


def get_kyoto_corpus(directory):
    url = 'http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz'
    filename = 'kftt-data-1.0.tar.gz'
    data_file = data_utils.maybe_download(directory, filename, url)
    if not (os.path.exists(os.path.join(directory, 'kyoto.ja')) and
            os.path.exists(os.path.join(directory, 'kyoto.en'))):
        print("Extracting file %s" % data_file)
        with tarfile.open(data_file, "r") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, directory)

        for lang in ['en', 'ja']:
            with open(os.path.join(directory, 'kyoto.%s' % lang), 'w', encoding='utf-8') as out_file:
                for dataset in ['train', 'dev', 'tune']:
                    in_path = os.path.join(data_file[:-7], 'data', 'orig', 'kyoto-%s.%s' % (dataset, lang))
                    with open(in_path, 'r', encoding='utf-8') as in_file:
                        out_file.write(in_file.read())


def get_enja_train_set(data_dir):
    if not (os.path.exists(os.path.join(data_dir, 'train.tok.ja')) and
            os.path.exists(os.path.join(data_dir, 'train.tok.en'))):

        if not os.path.exists(os.path.join(data_dir, 'tanaka')):
            os.mkdir(os.path.join(data_dir, 'tanaka'))
            get_tanaka_corpus(os.path.join(data_dir, 'tanaka'))
        if not os.path.exists(os.path.join(data_dir, 'kyoto')):
            os.mkdir(os.path.join(data_dir, 'kyoto'))
            get_kyoto_corpus(os.path.join(data_dir, 'kyoto'))

        for lang in ['en', 'ja']:
            out_path = os.path.join(data_dir, 'train.%s' % lang)
            with open(out_path, 'w', encoding='utf-8') as out_file:
                for dataset in ['tanaka', 'kyoto']:
                    in_path = os.path.join(data_dir, dataset, '%s.%s' % (dataset, lang))
                    with open(in_path, 'r', encoding='utf-8') as in_file:
                        out_file.write(in_file.read())

        # tokenization
        print("Preparing train data")
        os.system('%s -l en <%s >%s' % (en_tokenizer, os.path.join(data_dir, 'train.en'),
                                        os.path.join(data_dir, 'train.tok.en')))
        os.system('%s <%s >%s' % (ja_tokenizer, os.path.join(data_dir, 'train.ja'),
                                  os.path.join(data_dir, 'train.tok.ja')))

    return


def get_enja_dev_set(data_dir):
    if not (os.path.exists(os.path.join(data_dir, 'dev.tok.ja')) and
            os.path.exists(os.path.join(data_dir, 'dev.tok.en'))):
        get_kyoto_corpus(os.path.join(data_dir, 'kyoto'))
        for lang in ['en', 'ja']:
            in_file = os.path.join(data_dir, 'kyoto', 'kftt-data-1.0',
                                   'data', 'orig', 'kyoto-test.%s' % lang)
            shutil.copyfile(in_file, os.path.join(data_dir, 'dev.%s' % lang))

        # english tokenization
        print("Preparing dev data")
        os.system('%s -l en <%s >%s' % (en_tokenizer, os.path.join(data_dir, 'dev.en'),
                                        os.path.join(data_dir, 'dev.tok.en')))
        os.system('%s <%s >%s' % (ja_tokenizer, os.path.join(data_dir, 'dev.ja'),
                                  os.path.join(data_dir, 'dev.tok.ja')))

    return


def tokenizer(text):
    """We have done the tokenization step with external tools in advance."""
    return text.lower().split()



def prepare_data(data_dir, target_lang, source_lang,
                 target_vocab_size, source_vocab_size, download=False):
    """Get corpus into data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory in which the data sets will be stored.
      target_lang: Target language.
      source_lang: Source language.
      target_vocab_size: size of the target vocabulary to create and use.
      source_vocab_size: size of the source vocabulary to create and use.
      download: whether download the parallel corpora or not

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for target training data-set,
        (2) path to the token-ids for source training data-set,
        (3) path to the token-ids for target development data-set,
        (4) path to the token-ids for source development data-set,
        (5) path to the target vocabulary file,
        (6) path to the source vocabulary file.
    """
    # Get en-ja parallel corpora to the specified directory.
    if download:
        get_enja_train_set(data_dir)
        get_enja_dev_set(data_dir)

    train_path = os.path.join(data_dir, 'train.tok')
    dev_path = os.path.join(data_dir, 'dev.tok')

    # Create vocabularies of the appropriate sizes.
    target_vocab_path = os.path.join(data_dir, "vocab%d.%s" % (target_vocab_size, target_lang))
    source_vocab_path = os.path.join(data_dir, "vocab%d.%s" % (source_vocab_size, source_lang))
    data_utils.create_vocabulary(target_vocab_path, train_path + ".%s" % target_lang, target_vocab_size, tokenizer)
    data_utils.create_vocabulary(source_vocab_path, train_path + ".%s" % source_lang, source_vocab_size, tokenizer)

    # Create token ids for the training data.
    target_train_ids_path = train_path + (".ids%d.%s" % (target_vocab_size, target_lang))
    source_train_ids_path = train_path + (".ids%d.%s" % (source_vocab_size, source_lang))
    data_utils.data_to_token_ids(train_path + ".%s" % target_lang, target_train_ids_path, target_vocab_path, tokenizer)
    data_utils.data_to_token_ids(train_path + ".%s" % source_lang, source_train_ids_path, source_vocab_path, tokenizer)

    # Create token ids for the development data.
    target_dev_ids_path = dev_path + (".ids%d.%s" % (target_vocab_size, target_lang))
    source_dev_ids_path = dev_path + (".ids%d.%s" % (source_vocab_size, source_lang))
    data_utils.data_to_token_ids(dev_path + ".%s" % target_lang, target_dev_ids_path, target_vocab_path, tokenizer)
    data_utils.data_to_token_ids(dev_path + ".%s" % source_lang, source_dev_ids_path, source_vocab_path, tokenizer)

    return (target_train_ids_path, source_train_ids_path,
            target_dev_ids_path, source_dev_ids_path,
            target_vocab_path, source_vocab_path)