from utils import load_dataset_standard
from data_exploration import get_frequency_entities


def get_entities_(counts, new_dataset):
    """
    Extract the entity names

    :param counts: Counts of entities
    :param new_dataset: Main dataset
    :return: Entity names of each entity type
    """

    try:
        entity_dict = {k: [] for k in list(counts.keys())}

        for row in new_dataset['data']:
            entities = row['entity']
            text = row['primary_text']
            for entity in entities:
                entity_dict[entity[2]].append(text[entity[0]:entity[1]])
        return entity_dict
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in populating list of entities for every category: {0}'.format(e))


def top_entities(k, category, loc, dataset_id):
    """
    Shows the top k frequent entities for any category

    :param k: Upper limit of the required no. of entities per category.
    :param category: Required entity type/category.
    :param loc: Folder Location of working dataset
    :param dataset_id: Dataset Id
    :return: None
    """
    try:
        counts = get_frequency_entities.frequency_entities(loc, dataset_id, ret=True)
        ds = load_dataset_standard(loc, dataset_id)
        entity_words = get_entities_(counts, ds)
        # word_counts = {}
        words = entity_words[category]
        word_counts = {k: words.count(k) for k in words}
        top_words = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:k])
        print(top_words)
        return top_words
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in showing top entities for given category: {0}'.format(e))
