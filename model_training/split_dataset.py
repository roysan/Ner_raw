from sklearn.model_selection import train_test_split
from utils import load_dataset_standard
import spacy
from spacy.tokens import DocBin

nlp = spacy.load('en_core_web_sm')


def convert_to_spacy_binary_format(df):
    """
    Converts the dataset to spacy binary format for training/testing.
    :param df: Dataset in list of Tuples. <Text, (List of offset spans and labels) >
    :return: The binary format data
    """
    try:
        db = DocBin()
        for text, annotations in df:
            doc = nlp(text)
            ents = []
            for start, end, label in annotations:
                span = doc.char_span(start, end, label=label)
                ents.append(span)
            doc.ents = ents
            db.add(doc)
        return db
    except Exception as e:
        print('Error occurred in converting to binary format : {0}'.format(e))


def split_dataset_training(loc, dataset_id, dev_size, test_size, train_loc, dev_loc, test_loc):
    """
    Splits the main dataset into train, dev and test datasets.
    :param loc : Location of working dataset
    :param dataset_id: Dataset id
    :param dev_size: Split size for dev set
    :param test_size:Split size for test set
    :param train_loc: location to save train set in binary format
    :param dev_loc: location to save dev set in binary format
    :param test_loc: location to save test set in binary format
    :return:None
    """
    try:
        ds = load_dataset_standard(loc, dataset_id)
        test_total = int(test_size * len(ds['data']))
        dev_total = int(dev_size * len(ds['data']))

        text_entities = [[k['primary_text'], k['entity']] for k in ds['data']]

        temp, test_set = train_test_split(text_entities, test_size=test_total)
        train_set, dev_set = train_test_split(temp, test_size=dev_total)

        # Convert to binary format
        test_binary = convert_to_spacy_binary_format(test_set)
        dev_binary = convert_to_spacy_binary_format(dev_set)
        train_binary = convert_to_spacy_binary_format(train_set)

        print('Saving the split sets in location : {0},{1},{2}'.format(train_loc, dev_loc, test_loc))
        test_binary.to_disk(test_loc)
        dev_binary.to_disk(dev_loc)
        train_binary.to_disk(train_loc)

    except Exception as e:
        print('Error occurred in splitting main dataset : {0}'.format(e))
