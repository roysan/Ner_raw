import json
import os


def save_dataset(loc, dataset, dataset_id):
    """
    Function to save the working dataset in standard Weav format
    :param loc: Folder location to save the dataset
    :param dataset: Dataset to save
    :param dataset_id: Dataset Id
    :return: None
    """
    try:
        filename = 'standard_weav_format_' + str(dataset_id) + '.json'
        if os.path.isdir(loc):
            json.dump(dataset, open(os.path.join(loc, filename), 'w+'))
            print('Dataset saved in : {0} as {1}'.format(loc, filename))
        else:
            raise Exception('Output folder does not exist.')
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in loading labelled dataset: {0}'.format(e))


def load_dataset(loc, dataset_id):
    """
    Function to load the standard format dataset
    :param loc: Location of the dataset
    :param dataset_id: Dataset Id
    :return: Standard format dataset Loaded from folder
    """
    try:
        filename = 'standard_weav_format_' + str(dataset_id) + '.json'
        path = os.path.join(loc, filename)
        if os.path.exists(path):
            dataset = json.load(open(os.path.join(loc, filename), 'r+'))
            print('Dataset loaded in standard format from : {0} '.format(loc))
            return dataset
        else:
            raise Exception('File not found in location')
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Exception occured while loading the file with error : {0}'.format(e))


def token_entity_overlap(index, entities, text):
    """
    Check if there is an overlap of index of the word and all available entity spans
    :param index: Index of the word in the sentence
    :param entities: Entities and their offset spans
    :param text: Sentence text
    :return: Overlapping Entity and the actual label.
    """
    try:
        act_lab = act_text = None
        for st, ed, lab in entities:
            if st == index:
                act_lab = lab
                act_text = text[st:ed]
                break
            elif st <= index <= ed:
                act_lab = lab
                act_text = None
                break
        return act_lab, act_text
    except Exception as e:
        print('Error occurred in getting token entity overlap : {0}'.format(e))
