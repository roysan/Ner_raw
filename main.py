from data_loading import dataset_conversion, compile_data_unlabelled

from data_exploration import display_freq_distribution, get_frequency_entities
from data_exploration import get_top_entities, display_top_entities
from data_exploration import choose_entity
from data_exploration import get_word_distribution

from data_preprocessing import remove_misc, remove_punctuations
from data_preprocessing import use_regex, lemmatization, stemming
from data_preprocessing import case_normalization, remove_stopwords
from model_training import split_dataset
from model_training import convert_to_binary
from model_training import choose_config
from model_training import training, evaluate


# This unlabelled corpus is excluded in v2 of development
print('Compiling the total unlabelled data corpus for : Heath tweets data \n')
compile_data_unlabelled.compile_dataset_unlabelled('./datasets/Health-News-Tweets/Health-Tweets/',
                                                   './datasets/test_compiled')



print('Loading the labelled dataset :\n')
dataset_conversion.to_run(loc_input='./datasets/mit_movie_sample.txt',
                                          loc_output='./datasets', language='en', dataset_type='ner', dataset_id=1)

print('Display the frequency of categorical entities in dataset :\n')
get_frequency_entities.frequency_entities(loc='./datasets', dataset_id=1, ret=False)

print('Display the graph of categorical entities :\n')
display_freq_distribution.frequency_distribution(loc='./datasets/', dataset_id=1)

print('Display top k entities per user choice for category with counts :\n')
get_top_entities.top_entities(k=5, category='ACTOR', loc='./datasets/', dataset_id=1)

print('Display top entities of chosen category :\n')
display_top_entities.display_top_entities(k=5, category='ACTOR', loc='./datasets/', dataset_id=1)

print('Display word distribution in dataset : \n')
get_word_distribution.word_distribution(loc='./datasets/', dataset_id=1)

print('Choose the entity to work with :\n')
choose_entity.choose_entity(entity_list=['ACTOR', 'GENRE'], loc='./datasets/', dataset_id=1, loc_output='./datasets/')

# Data preprocessing

print('Use lemmatization to use lemmas : \n')
lemmatization.lemmatization(loc='./datasets/', dataset_id=1)

print('Use stemming to use stem of words : \n')
stemming.stemming(loc='./datasets/', dataset_id=1)

print('Use case normalization - Sentence case :\n')
case_normalization.case_normalize(loc='./datasets/', dataset_id=1, case='sentence')

print('Remove stopwords : \n')
remove_stopwords.remove_stopwords(loc='./datasets/', dataset_id=1)

print('Remove punctuations - [!,.] : \n')
remove_punctuations.remove_punctuations(loc='./datasets/', dataset_id=1, punct='!,.')

print('Remove miscellaneous characters -  [@, \U0001F923] : \n')
remove_misc.remove_misc(loc='./datasets/', dataset_id=1, chars=['@', '\U0001F923'])

print('Using regex : office* to replace : Algo  in dataset : \n')
use_regex.sub_regex(loc='./datasets/', dataset_id=1,
                    regex='office*', substitution=True, substitute='Algo')

# Model training

print('Split the dataset into train dev test as per ratio 70-15-15 : \n')
split_dataset.split_dataset_training('./datasets/working_temp.txt', 0.15, 0.15,
                                     './datasets/train_spacy_sample.txt', './datasets/dev_spacy_sample',
                                     './datasets/test_spacy_sample')

print('Convert to binary format for spacy v3.0 : \n')
convert_to_binary.convert_to_spacy_binary_format('./datasets/train_spacy_sample.txt',
                                                 './datasets', 'iob', n_sents=None, seg_sents=None)
convert_to_binary.convert_to_spacy_binary_format('./datasets/dev_spacy_sample.txt',
                                                 './datasets', 'iob', n_sents=None, seg_sents=None)
convert_to_binary.convert_to_spacy_binary_format('./datasets/test_spacy_sample.txt',
                                                 './datasets', 'iob', n_sents=None, seg_sents=None)

print('Choose the config file according to dataset id, architecture and embedding type \n')
choose_config.choose_config(dataset_id=1, architecture='transformers')
# choose_config.choose_config(dataset_id=1, architecture='cnn', embedding_type = 'char')


print('Start Model training : \n')
training.train('./config_files/test_config.cfg', './output_spacy', gpu_id='-1',
               train_path='./datasets/train_spacy_sample.spacy', dev_path='./datasets/dev_spacy_sample.spacy',
               dropout='0.1', max_epochs='5', max_length='2000', eval_frequency='200', batch_size='1000')

print('Evaluate model : \n')
evaluate.evaluate(model_path='./output_spacy/model-best/', test_path='./datasets/test_spacy_sample.spacy',
                  output_file=None)
