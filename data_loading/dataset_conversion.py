from spacy.training import iob_to_biluo
from spacy.training import biluo_tags_to_offsets
from utils import save_dataset
import spacy

nlp = spacy.load('en_core_web_sm')


def modify_labels(text, labels):
    """
    Modify the loaded labels and convert them according to spacy's tokenized distribution (Change the distribution to
    meet spacy's tokenization rules and offset spans)
    :param text: Primary text
    :param labels: The existing loaded labels
    :return: new_labels: The modified labels
    """
    try:
        new_labels = []
        for tok, lab in zip(text.split(' '), labels):
            doc = nlp(tok)
            if len(doc) > 1:
                extra_token_len = len(doc) - 1
                if lab.startswith('B'):
                    new_lab = ['I-' + lab.split('-')[-1]] * extra_token_len
                else:
                    new_lab = [lab] * extra_token_len
                new_labels.append(lab)
                new_labels.extend(new_lab)
            else:
                new_labels.append(lab)
        assert len(nlp(text)) == len(new_labels)
        return new_labels
    except Exception as e:
        print('Error occurred in modifying labels : {0}'.format(e))


def dataset_conversion(loc_input, loc_output, language='en', dataset_type='ner', dataset_id=1):
    """
    Load the IOB labelled data from mit movie corpus and convert to weav standard format.
    :param loc_input: Location of the labelled dataset. (mit movie corpus)
    :param loc_output: Folder Location to save the main dataset in standard Weav AI format.
    :param language: language of the text
    :param dataset_type: Type of dataset in use
    :param dataset_id: Id of dataset passed.
    :return: None
    """
    new_dataset = {}
    sent_id = 0
    tokens = list()
    labels = list()
    final = dict()
    final['dataset_id'] = dataset_id
    final['dataset_type'] = dataset_type
    final['data'] = []
    new_text = ''
    offset_spans = []
    try:
        with open(loc_input, 'r') as f:
            k = f.readlines()
        for line in k:
            line = line.strip()
            if line != '':
                token, label = line.split('\t')
                tokens.append(token)
                labels.append(label)
            elif len(tokens) > 0:
                assert len(tokens) == len(labels)
                new_dataset[sent_id] = {'tokens': tokens, 'labels': labels}
                # convert iob tags to offset spans as standard format
                new_text = ' '.join(tokens)
                doc = nlp(' '.join([str(x) for x in nlp(new_text)]))
                if len(doc) != len(labels):
                    labels = modify_labels(new_text, labels)
                biluo_labels = iob_to_biluo(labels)
                offset_spans = biluo_tags_to_offsets(doc, biluo_labels)
                new_string = ' '.join([str(x) for x in doc])
                final['data'].append({'sentence_id': sent_id, 'primary_text': new_string,
                                      'primary_label': None, 'primary_language': language, 'secondary_text': None,
                                      'secondary_language': None, 'entity': offset_spans,
                                      'similarity_score': None})
                sent_id += 1
                tokens = list()
                labels = list()
        if tokens:
            new_dataset[sent_id] = {'tokens': tokens, 'labels': labels}
            final['data'].append({'sentence_id': sent_id, 'primary_text': new_text,
                                  'primary_label': None, 'primary_language': language, 'secondary_text': None,
                                  'secondary_language': None, 'entity': offset_spans,
                                  'similarity_score': None})

        save_dataset(loc_output, final, dataset_id)

    except FileNotFoundError:
        print('Path not found to load the file')
    except ValueError:
        print('Values are missing in the labelled dataset')
        print('Please try again with reformatted data.')
    except Exception as e:
        print('Error occurred in converting dataset to standard format: {0}'.format(e))


def to_run(loc_input, loc_output, language='en', dataset_type='ner', dataset_id=1):
    dataset_conversion(loc_input, loc_output, language, dataset_type, dataset_id)
