from utils import load_dataset_standard, save_dataset_standard
from utils import token_entity_overlap
import spacy
import re

nlp = spacy.load('en_core_web_sm')


def sub_regex(loc, dataset_id, regex=None, substitution=False, removal=False, substitute=None):
    """
    To use custom regex function to either substitute or remove from text
    :param substitute: The substitute value in case of substitution
    :param regex: The regular expression pattern to use
    :param loc: Location of working dataset
    :param dataset_id: Dataset Id
    :param substitution: Whether to substitute
    :param removal: Whether to remove
    :return: None
    """

    try:
        nt = ''
        ds = load_dataset_standard(loc, dataset_id)
        for row in ds['data']:
            text = row['primary_text']
            entities = row['entity']
            new_tok = []
            new_entities = []

            for word in nlp(text):
                lab, ntt = token_entity_overlap(word.idx, entities, text)

                if not lab and not ntt:
                    if substitution:
                        nt = re.sub(regex, substitute, str(word))
                    elif removal:
                        nt = re.sub(regex, '', str(word))
                    if nt != '':
                        new_tok.append(nt)
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
        print('Error occurred in using custom regex : {0}'.format(e))
