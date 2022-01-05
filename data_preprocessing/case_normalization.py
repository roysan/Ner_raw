from utils import load_dataset_standard, save_dataset_standard
import spacy

nlp = spacy.load('en_core_web_sm')


def case_normalize(loc, dataset_id, case='lower'):
    """
    Use case normalization as per user choice
    :param loc: Location of working dataset
    :param dataset_id: Dataset Id
    :param case: Which case to normalize. (Lower, Upper, Sentence)
    :return: None
    """

    try:
        ds = load_dataset_standard(loc, dataset_id)
        for row in ds['data']:
            text = row['primary_text']

            if case == 'lower':
                row['primary_text'] = ' '.join([str(k).lower() for k in nlp(text)])
            elif case == 'upper':
                row['primary_text'] = ' '.join([str(k).upper() for k in nlp(text)])
            elif case == 'sentence':
                row['primary_text'] = ' '.join([str(nlp(text)[0]).capitalize()]
                                               + [str(k).lower() for k in nlp(text)[1:]])

        save_dataset_standard(loc, ds, dataset_id)
        print('Altered dataset saved successfully!')
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print('Error occurred in getting case normalization : {0}'.format(e))
