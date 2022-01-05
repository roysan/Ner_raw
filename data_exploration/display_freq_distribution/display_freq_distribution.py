import seaborn as sns
import matplotlib.pyplot as plt
from data_exploration import get_frequency_entities


def frequency_distribution(loc, dataset_id):
    """
    Display frequency distribution of all entity categories
    :param dataset_id: Dataset Id
    :param loc: Location of the working dataset
    :return: None
    """

    try:
        counts = get_frequency_entities.frequency_entities(loc, dataset_id, ret=True)
        # Sort the items in descending order
        counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        keys = list(counts.keys())
        values = list(counts.values())

        pl = sns.barplot(x=keys, y=values)
        pl.set_title('Frequency of entities per category')
        pl.set_xlabel('Category')
        pl.set_ylabel('Count')
        plt.show()
        return plt
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in displaying frequency distribution of entities: {0}'.format(e))
