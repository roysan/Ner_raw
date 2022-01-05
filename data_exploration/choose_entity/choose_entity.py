from utils import load_dataset_standard, save_dataset_standard


def choose_entity(entity_list, loc, dataset_id, loc_output):
    """
    User choice for the entities to be trained. Removes the remaining entities from labels and saves the new dataset.
    :param entity_list: List of entities to be trained
    :param loc: Folder Location of working dataset
    :param dataset_id: Dataset Id
    :param loc_output: Folder location to save modified dataset
    :return: None
    """
    try:
        ds = load_dataset_standard(loc, dataset_id)
        for row in ds['data']:
            entities_to_remove = [ntt for ntt in row['entity'] if ntt[2] not in entity_list]
            [row['entity'].remove(ntt) for ntt in entities_to_remove]

        save_dataset_standard(loc_output, ds, dataset_id)
        print('Saving dataset with only following entities : {0}\n'.format(entity_list))

    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in getting word length distribution : {0}'.format(e))
