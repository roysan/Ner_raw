import os
import sys
from utils import save_dataset


def compile_dataset_unlabelled(loc_input, loc_output):
    """
    Function to get the complete compiled unlabelled dataset as per user file location & save a compiled set
    :param loc_input: Location of the downloaded dataset.
    :param loc_output: Location where to save the compiled dataset.
    :return: The compiled data in the form of a single dataset.
    """

    try:
        tweets_full = list()
        files = os.listdir(loc_input)
        for file_ in files:
            with open(os.path.join(loc_input, file_), 'rb') as f:
                k = f.readlines()
                # binary conversion to string ignoring possible errors
                tweets = [line.strip().decode('utf-8', errors='ignore') for line in k]
            tweets_full.extend(tweets)
        print('Dataset compiled !')
        save_dataset(loc_output, tweets_full)
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Data location invalid.')
