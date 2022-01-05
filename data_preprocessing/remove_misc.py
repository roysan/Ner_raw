from utils import load_dataset_standard, save_dataset_standard
from utils import token_entity_overlap
import spacy

nlp = spacy.load('en_core_web_sm')


def remove_misc(loc, dataset_id, chars):
    """
    To remove miscellaneous characters from the text chunks as user choice
    :param chars: List of Characters to remove as per user choice
    :param loc: Location of working dataset
    :param dataset_id: Dataset Id
    :return: None
    """

    try:
        ds = load_dataset_standard(loc, dataset_id)
        for row in ds['data']:
            text = row['primary_text']
            entities = row['entity']
            new_tok = []
            new_entities = []
            for word in nlp(text):
                lab, ntt = token_entity_overlap(word.idx, entities, text)

                if not lab and not ntt:
                    if all([str(word) not in char for char in chars]):
                        new_tok.append(str(word))
                elif lab and not ntt:
                    new_tok.append(str(word))
                else:
                    current_idx = len(' '.join(new_tok))
                    if current_idx == 0:
                        current_idx = -1
                    current_entity = [current_idx + 1, current_idx + 1 + len(ntt), lab]
                    new_entities.append(current_entity)
                    new_tok.append(str(word))

            text_new = ' '.join(new_tok)
            row['primary_text'] = text_new
            row['entity'] = new_entities

        save_dataset_standard(loc, ds, dataset_id)
        print('Altered dataset saved successfully!')

    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in remove miscellaneous tokens : {0}'.format(e))
