from utils import load_dataset_standard, save_dataset_standard
from nltk.stem.porter import PorterStemmer
from utils import token_entity_overlap
import spacy

nlp = spacy.load('en_core_web_sm')
porter_stemmer = PorterStemmer()


def stemming(loc, dataset_id):
    """
    Use stemming to get to the root of the words/tokens
    :param loc: Location of working dataset
    :param dataset_id: Dataset Id
    :return: None
    """

    try:
        ds = load_dataset_standard(loc, dataset_id)
        for row in ds['data']:
            new_tok = []
            new_entities = []
            text = row['primary_text']
            entities = row['entity']
            if not entities:
                continue
            for word in nlp(text):
                lab, ntt = token_entity_overlap(word.idx, entities, text)
                if not lab and not ntt:
                    new_tok.append(porter_stemmer.stem(str(word)))
                elif lab and ntt:
                    current_idx = len(' '.join(new_tok))
                    if current_idx == 0:
                        current_idx = -1
                    current_entity = [current_idx + 1, current_idx + 1 + len(ntt), lab]
                    new_entities.append(current_entity)
                    new_tok.append(str(word))
                else:
                    new_tok.append(str(word))

            text_lemmatized = ' '.join(new_tok)

            row['primary_text'] = text_lemmatized
            row['entity'] = new_entities

        save_dataset_standard(loc, ds, dataset_id)
        print('Altered dataset saved successfully!')
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in getting stemmed tokens : {0}'.format(e))
