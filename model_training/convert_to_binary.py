import subprocess


def convert_to_spacy_binary_format(input_file, output_dir, converter, n_sents=None, seg_sents=None):
    """
    Convert the dataset to the spacy 3.0 binary format
    :param converter: Type of converter to be used for specific format
    :param input_file: Input file required in specific format : iob, biluo, conll
    :param output_dir: Directory to save the output file
    :param n_sents: No of sentences per document.
    :param seg_sents: Whether to segment the sentences.
    :return: Binary format dataset
    """
    try:
        if n_sents and seg_sents:
            subprocess.run(
                ["python", "-m", "spacy", "convert", input_file, output_dir, '-c', converter, "-n", n_sents, '-s'])
        elif n_sents:
            subprocess.run(
                ["python", "-m", "spacy", "convert", input_file, output_dir, '-c', converter, "-n", n_sents])
        elif seg_sents:
            subprocess.run(
                ["python", "-m", "spacy", "convert", input_file, output_dir, '-c', converter, '-s'])
        else:
            subprocess.run(
                ["python", "-m", "spacy", "convert", input_file, output_dir, '-c', converter])
        print('Generated spacy binary format file')
    except Exception as e:
        print('Error occurred in conversion to binary format : {0}'.format(e))

