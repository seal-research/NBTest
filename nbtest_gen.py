import os
import os
import nbformat as nbf
import argparse
import json
import sys
from pathlib import Path
import shutil

proj_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_folder)
from properties.propertyFinder import PropertyFinder
from assert_gen.notebookRunner import NotebookRunner
from assert_gen.assertionGenerator import AssertionGenerator
from utils.pytest_run import run_missing_tests, aggregate_results_to_csv
from utils.utils import setup_logger, get_notebook_name

config_path = os.path.join(proj_folder, "config.json")
global CONFIG_FILEPATH

with open(config_path, "r") as config_file:
    CONFIG_FILEPATH = json.load(config_file)["FILE_PATH"]

def main():
    parser = argparse.ArgumentParser(description='Tool for generating tests for Jupyter Notebooks')
    parser.add_argument('notebook', type=str, help='The path to the Jupyter notebook file')
    parser.add_argument('-n', default="30", help='Number of iterations for generating assertions (default: 30)')
    parser.add_argument('-o', required=True, help='Absolute path for output files')
    parser.add_argument("-m", default="chebyshev", choices=["hoeffding", "chebyshev"], help="Method for bound calculation.")
    parser.add_argument('-c', default=0.95, help='Confidence level for the bounds calculation (default: 0.95)')
    parser.add_argument("--pytest", action="store_true", help="Run pytest with nbtest plugin to test the assertions in the notebook")
    parser.add_argument('-pn', default="1", help='Number of iterations for running the notebook when --pytest is provided')
    parser.add_argument('--rundir', default='./', help='Folder to run the notebook. This folder should have the file structure required for the notebook')
    args = parser.parse_args()

    global stats_method, conf_level, output_dir, iterations, notebook_path
    stats_method = args.m
    conf_level = float(args.c)
    output_dir = os.path.abspath(args.o)

    iterations = int(args.n)
    pytest_iterations = int(args.pn)
    notebook_path = os.path.abspath(args.notebook)

    rundir = os.path.abspath(args.rundir)
    original_nb_folder = Path(notebook_path).parent
    os.makedirs(output_dir, exist_ok=True)

    # Paths
    notebook_fname = get_notebook_name(notebook_path)

    tmp_ipynb = os.path.join(original_nb_folder, f"{notebook_fname}_{CONFIG_FILEPATH['TMP_IPYNB']}")
    instrumentation_pkl = os.path.join(output_dir, f"{notebook_fname}_{CONFIG_FILEPATH['INSTRUMENTATION_PKL']}")
    properties_csv = os.path.join(output_dir, f"{notebook_fname}_{CONFIG_FILEPATH['PROPERTIES_CSV']}")

    if args.pytest:
        logs_dir = os.path.join(output_dir, f"{notebook_fname}_logs")
        os.makedirs(logs_dir, exist_ok=True)
        logger = setup_logger(os.path.join(logs_dir, f'main.log'), 'main')
        if os.path.exists(notebook_path):
            # Run pytest over notebooks to test assertions
            run_missing_tests(output_dir, notebook_path, pytest_iterations, rundir)
            aggregate_results_to_csv(output_dir, notebook_path)
        else:
            logger.error(f"{notebook_path} doesn't exist.")

    else:
        original_copy_ipynb = os.path.join(output_dir, f"{notebook_fname}.ipynb")
        shutil.copy(notebook_path, original_copy_ipynb)
        # Generate assertions for notebooks

        # 1. Find metrics and generate assertions

        property_finder = PropertyFinder(notebook_path, output_dir, original_nb_folder)
        property_finder.run()

        # 2. Run notebook to collect values used in assertions
        runner = NotebookRunner(
            notebook_path=tmp_ipynb,
            iterations=iterations,
            output_dir=output_dir
        )
        runner.run()

        # 3. Generate assertions based on the properties and values collected

        generator = AssertionGenerator(
            instrumentation_pkl=instrumentation_pkl,
            original_copy_ipynb=original_copy_ipynb,
            properties_csv=properties_csv,
            output_dir=output_dir,
            notebook_fname=notebook_fname,
            stats_method=stats_method,
            conf_level = conf_level,
            iteration = iterations
        )
        generator.run()

if __name__ == "__main__":
    sys.exit(main())
