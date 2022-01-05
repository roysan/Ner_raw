import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_dataset_standard


def word_distribution(loc, dataset_id):
    """
    To show the distribution of words and entities in each text chunk with plots.
    :param loc: Location of working dataset.
    :param dataset_id:Dataset Id
    :return: Density plot for text chunk length.
    """
    try:
        ds = load_dataset_standard(loc, dataset_id)
        dist = [len(x['primary_text'].split(' '))  for x in ds['data']]
        mean = np.mean(dist)
        minim = np.min(dist)
        maxim = np.max(dist)
        std = np.std(dist)
        median = np.median(dist)
        print('Distribution of tokens in text chunks : \n')
        print('Minimum words : ', minim)
        print('Maximum words : ', maxim)
        print('Average words : ', mean)
        print('Median words : ', median)
        print('Standard deviation : ', std)

        pl = sns.distplot(dist)
        pl.set_title('Density Plot for text chunk lengths')
        pl.set_xlabel('Sentence Lengths')
        plt.show()
        return pl
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in getting word length distribution : {0}'.format(e))