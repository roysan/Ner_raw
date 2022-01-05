import subprocess


def set_config(output_file, optimize='accuracy', gpu='', pipeline='ner'):
    """
    Set the configuration for model training.
    :param output_file: Location of output file
    :param optimize: Optimize the code for accuracy or efficiency
    :param gpu: Whether gpu is needed or not
    :param pipeline: List of pipeline including NER
    :return: None
    """
    try:
        if gpu:
            subprocess.run(["python", "-m", "spacy", "init", "config", output_file, "--pipeline", pipeline,
                            "--optimize", optimize, gpu, '-F'])
        else:
            subprocess.run(["python", "-m", "spacy", "init", "config", output_file, "--pipeline", pipeline,
                            "--optimize", optimize, '-F'])

        print('Config built successfully!')
    except Exception as e:
        print('Error occurred in filling config file : {0}'.format(e))


def fill_config(base_path, output_file):
    """
    Auto Fill a partially filled config file with all default values.
    :param base_path: Path to the partially filled config file
    :param output_file: Output file path.
    :return: None
    """
    try:
        subprocess.run(["python", "-m", "spacy", "init", "fill-config", base_path, output_file, '--diff'])
    except Exception as e:
        print('Error occurred in auto filling config: {0}'.format(e))
