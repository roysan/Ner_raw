import matplotlib.pyplot as plt
import seaborn as sns
from data_exploration import get_top_entities


def display_top_entities(k, category, loc, dataset_id):
    """
    Display frequency distribution of top entities by category
    :param k: Upper limit of the required no. of entities per category.
    :param category: Entity category
    :param loc: Folder Location of working dataset
    :param dataset_id: Dataset Id
    :return: Plot showing the top k entities
    """
    try:
        top_words = get_top_entities.top_entities(k, category, loc, dataset_id)
        keys = list(top_words.keys())
        values = list(top_words.values())

        pl = sns.barplot(x=keys, y=values)
        pl.set_title('Top entities for {0} category'.format(category))
        pl.set_xlabel('Entities')
        pl.set_ylabel('Count')
        plt.show()
        return pl
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in displaying top entities for given category: {0}'.format(e))