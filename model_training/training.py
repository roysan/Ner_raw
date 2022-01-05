import subprocess


def train(config_path, output_path, gpu_id='-1', train_path=None, dev_path=None, dropout='0.1', max_epochs='10',
          max_length='2000', eval_frequency='200', batch_size='1000', vectors='null', include_vectors='False',
          use_transformers=False, transformer_model='null'):
    """
    Main training function
    :param transformer_model:
    :param use_transformers:
    :param config_path: Path to config file
    :param output_path: Path where to store the training output
    :param gpu_id: id of GPU. -1 is default for CPU.
    :param train_path: Path to train set in binary spacy format
    :param dev_path: Path to dev set in binary spacy format
    :param dropout: Dropout rate. Default is 0.1
    :param max_length: Maximum length of tokens to be considered
    :param eval_frequency: Frequency at which the evaluation on dev set is calculated
    :param max_epochs: Max epochs to train
    :param batch_size: Batch size of the training data
    :param vectors: Pretrained static vectors
    :param include_vectors: Whether to use the static vectors or not
    :return: None
    """
    try:
        # Get the training running in CLI:

        if use_transformers:
            if transformer_model == 'null':
                raise Exception('Please enter a valid Transformer model.'
                                'For details please check : https://huggingface.co/models')

            subprocess.run(["python", "-m", "spacy", "train", config_path, '--output',
                            output_path, "--gpu-id", gpu_id, '--paths.train', train_path, '--paths.dev', dev_path,
                            '--training.dropout', dropout, '--training.max_epochs', max_epochs,
                            '--corpora.train.max_length', max_length, '--corpora.dev.max_length', max_length,
                            '--training.eval_frequency', eval_frequency, '--nlp.batch_size', batch_size,
                            '--paths.transformer_model', transformer_model])
        else:
            if vectors == 'null':
                subprocess.run(["python", "-m", "spacy", "train", config_path, '--output',
                                output_path, "--gpu-id", gpu_id, '--paths.train', train_path, '--paths.dev', dev_path,
                                '--training.dropout', dropout, '--training.max_epochs', max_epochs,
                                '--corpora.train.max_length', max_length, '--corpora.dev.max_length', max_length,
                                '--training.eval_frequency', eval_frequency, '--nlp.batch_size', batch_size,
                                ])
            else:
                subprocess.run(["python", "-m", "spacy", "train", config_path, '--output',
                                output_path, "--gpu-id", gpu_id, '--paths.train', train_path, '--paths.dev', dev_path,
                                '--training.dropout', dropout, '--training.max_epochs', max_epochs,
                                '--corpora.train.max_length', max_length, '--corpora.dev.max_length', max_length,
                                '--training.eval_frequency', eval_frequency, '--nlp.batch_size', batch_size,
                                '--paths.use_static_vectors', include_vectors, '--paths.vectors', vectors])

    except Exception as e:
        print('Error Occurred in training due to : {0}'.format(e))
