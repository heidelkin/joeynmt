# coding: utf-8
"""
Data module
"""
import sys
import os
import os.path
from typing import Optional

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary


def load_data(data_cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]

    ## changed to allow no dev_path
    dev_path = data_cfg.get("dev", None)
    feedback_suffix = data_cfg.get("feedback", None)
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    # use feedback information as well
    if feedback_suffix is not None:
        weight_field = data.RawField()
        # token or sentence weights are given for training target
        train_data = WeightedTranslationDataset(
            level=level,
            path=train_path,
            exts=("." + src_lang, "." + trg_lang, "." + feedback_suffix),
            fields=(src_field, trg_field, weight_field),
            filter_pred=
            lambda x: len(vars(x)['src']) <= max_sent_length and
                      len(vars(x)['trg']) <= max_sent_length)

    else:
        train_data = TranslationDataset(path=train_path,
                                        exts=("." + src_lang, "." + trg_lang),
                                        fields=(src_field, trg_field),
                                        filter_pred=
                                        lambda x: len(vars(x)['src'])
                                        <= max_sent_length
                                        and len(vars(x)['trg'])
                                        <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)

    # modified for no dev set cases
    dev_data = None
    if dev_path is not None:
        if os.path.isfile(dev_path + "." + trg_lang):
            dev_data = TranslationDataset(path=dev_path,
                                          exts=("." + src_lang, "." + trg_lang),
                                          fields=(src_field, trg_field))
        else:
            dev_data = MonoDataset(path=dev_path, ext="." + src_lang,
                                   field=src_field)

    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab


def make_data_iter(dataset: Dataset, batch_size: int, train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """
    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.Iterator(
            repeat=False, dataset=dataset, batch_size=batch_size,
            train=False, sort=False)

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)


class WeightedTranslationDataset(Dataset):
    """ Defines a parallel dataset with weights for the targets. """

    def __init__(self, path, exts, fields, level, **kwargs):
        """Create a TranslationDataset given paths and fields.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        :param level: char or word or bpe
        """

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]),
                      ('weights', fields[2])]

        src_path, trg_path, feedback_path = tuple(os.path.expanduser(path + x)
                                                  for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file, \
                open(feedback_path) as feedback_file:
            for src_line, trg_line, weights_line in \
                    zip(src_file, trg_file, feedback_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                weights = [float(weight) for weight in
                           weights_line.strip().split(" ")]
                if src_line != '' and trg_line != '':
                    # there must be feedback for every token
                    if level == "char":
                        char_weights = []
                        # distribute feedback from tokens over chars
                        assert len(trg_line.split()) == len(weights)
                        for trg_token, token_weight in zip(trg_line.split(),
                                                           weights):
                            # replicate weight for every char in trg token
                            # and for following whitespace
                            char_weights.extend(
                                (len(trg_token)+1)*[token_weight])
                        # remove last added weight for whitespace
                        weights = char_weights[:-1]
                    assert len(weights) == len(fields[1][1].tokenize(trg_line))
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line, weights], fields))

        super(WeightedTranslationDataset, self).__init__(examples,
                                                         fields, **kwargs)
