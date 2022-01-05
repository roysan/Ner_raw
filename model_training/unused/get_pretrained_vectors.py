import subprocess


def get_pretrained_vectors(vectors_loc, output_dir, name, prune='-1'):
    """
    Load/initialize the pretrained vectors/embeddings to the training pipeline. Eg. (en_core_sci_md, en_core_web_md has
    string.json files inside vocab dir)
    :param prune: Number of vectors to prune the vocab to.
    :param vectors_loc: Location of the pretrained vectors vocab files.
    :param output_dir: Location to save the vectors after adding to pipeline.
    :param name: Name to be given to the vectors.
    :return: None
    """
    try:
        subprocess.run(["python", "-m", "spacy", "init", "vectors", "en", vectors_loc, output_dir, "--name",
                        name, '--prune', prune])
        print('Vectors saved in {0}'.format(output_dir))
    except Exception as e:
        print('Error occurred in auto filling config: {0}'.format(e))
