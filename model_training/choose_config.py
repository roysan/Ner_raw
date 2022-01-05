import shutil
import os


def choose_config(dataset_id, architecture='cnn', embedding_type='word'):
    """
    Select the architecture configuration according to user choice.
    :param dataset_id: Dataset id
    :param architecture: Architecture for algorithm (Either cnn, lstm or transformers)
    :param embedding_type: Whether to use character or word embedding. Not applicable for transformers (Either
    :return: None
    """
    try:
        if embedding_type not in ['word', 'char']:
            raise Exception("Embedding type values include 1. word 2. char")
        if architecture == 'transformers':
            filename = 'config_transformers.cfg'
        elif architecture in ['cnn', 'lstm']:
            filename = 'config_' + str(embedding_type) + '_embed_' + str(architecture) + '_encode.cfg'
        else:
            raise Exception('Supported Architectures can be of 3 types : 1. transformers 2. cnn 3. lstm')
        file_loc = os.path.join('./config_files', filename)
        if not os.path.exists(file_loc):
            raise Exception("Config File not found in location. Please check!")
        else:
            work_file_loc = './config_files/config_{0}.cfg'.format(dataset_id)
            shutil.copy(file_loc, work_file_loc)
        shutil.copy(file_loc, work_file_loc)
        print('Config file for dataset id : {0} saved in : {1}'.format(dataset_id, work_file_loc))

    except Exception as e:
        print('Error occurred in choosing config file. : {0}'.format(e))
