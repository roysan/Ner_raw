from utils import load_dataset_standard


def frequency_entities(loc, dataset_id, ret=True):
    """
    Get the categorical count of categories
    :param loc: The location of dataset to work with.
    :param dataset_id: Dataset Id
    :param ret: Whether to return a value or not
    :return: Count of entities.
    """

    try:
        new_dataset = load_dataset_standard(loc, dataset_id)
        labels_ = [entity[2] for row in new_dataset['data'] for entity in row['entity']]
        unique_labels = list(set(labels_))
        counts = {c: labels_.count(c) for c in unique_labels}

        if ret:
            return counts
        else:
            print('Categorical count of entities : \n', counts)
    except ValueError:
        print('Some error occurred during counting of the values. Empty value found.')
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in getting frequency of entities: {0}'.format(e))
