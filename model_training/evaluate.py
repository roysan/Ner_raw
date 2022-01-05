import subprocess


def evaluate(model_path, test_path, output_file=None, gpu_id='-1'):
    """
    Main function to evaluate model on test data
    :param model_path: Path to the trained spacy model
    :param test_path: Path to test file for evaluation
    :param output_file: Path to save the output results of evaluation.
    :param gpu_id: id of GPU. -1 is default for CPU.
    :return: None
    """
    try:
        # Get the evaluation running in CLI:
        if output_file:
            subprocess.run(["python", "-m", "spacy", "evaluate", model_path, test_path,
                            '--output', output_file, "--gpu-id", gpu_id])
        else:
            subprocess.run(["python", "-m", "spacy", "evaluate", model_path, test_path, "--gpu-id", gpu_id])
    except Exception as e:
        print('Error Occurred in evaluation due to : {0}'.format(e))
