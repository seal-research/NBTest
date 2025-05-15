import os
import glob
import re
import subprocess
import pandas as pd
import logging
import sys
from pathlib import Path
import shutil
from pathlib import Path

proj_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(proj_folder)

from utils.utils import setup_logger


# Collect iterations already run
def get_completed_iterations(output_dir, notebook_fname):

    pattern = os.path.join(output_dir, f'{notebook_fname}_pytest*.txt')
    return [
        int(match.group(1)) for file_path in glob.glob(pattern)
        if (match := re.search(r'pytest(\d+)\.txt', file_path))  # Match .txt files
    ]


# Run missing tests
def run_missing_tests(output_dir, notebook_path, pytest_iterations, run_folder):
    notebook_stat_name = Path(notebook_path).stem
    notebook_fname = notebook_stat_name.split("_chebyshev")[0]

    is_copy = False

    if run_folder is None:
        run_folder = os.path.dirname(notebook_path)
    else:
        if os.path.dirname(notebook_path) != run_folder:
            original_copy_ipynb = os.path.join(run_folder, f"{notebook_stat_name}.ipynb")
            if os.path.abspath(notebook_path) != os.path.abspath(original_copy_ipynb):

                shutil.copy(notebook_path, original_copy_ipynb)
                # if os.path.exists(original_copy_ipynb):
                #     print(f"Copying {notebook_path} to {original_copy_ipynb}")
                is_copy = True


            notebook_path = original_copy_ipynb

    prev_ite = get_completed_iterations(output_dir, notebook_fname)

    for i in range(1, pytest_iterations + 1):
        if i not in prev_ite:
            iter_txt_path = os.path.join(output_dir, f'{notebook_fname}_pytest{i}.txt')

            logs_dir = os.path.join(output_dir, f"{notebook_fname}_logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_file_path = os.path.join(logs_dir, f'{notebook_fname}_pytest{i}.log')

            logger = setup_logger(log_file_path, "iteration_{i}")

            logger.info(f"Starting iteration {i}...")

            logger.info(f"Running pytest on {notebook_path}")

            result = subprocess.run(
                ["pytest", "--nbtest", "-v", notebook_path,
                "--nbtest-output-dir", output_dir,
                "--nbtest-log-filename", os.path.join(output_dir, f"{notebook_fname}_pytest_testid_res_{i}.csv"),
                "--nbtest-nblog-name", os.path.join(output_dir, f"{notebook_fname}_pytest_nb_tracebacks_{i}.ipynb")],
                capture_output=True,
                text=True,
                cwd=run_folder
            )

            logger.info(f"STDOUT:\n{result.stdout}")
            logger.debug(f"STDERR:\n{result.stderr}")

            status = "Fail" if result.returncode not in (0, 5) else "Pass"

            with open(iter_txt_path, "w") as file:
                file.write(status)
    if is_copy:
        Path(notebook_path).unlink() # Delete file


# Aggregate results into a CSV file
def aggregate_results_to_csv(output_dir, notebook_path):

    notebook_stat_name = Path(notebook_path).stem
    notebook_fname = notebook_stat_name.split("_chebyshev")[0]

    completed_iterations = get_completed_iterations(output_dir, notebook_fname)

    rows = []
    total_pass = 0
    total_fail = 0
    for iteration in completed_iterations:
        iter_txt_path = os.path.join(output_dir, f'{notebook_fname}_pytest{iteration}.txt')
        try:
            with open(iter_txt_path, "r") as file:
                status = file.read().strip()
                rows.append({
                    "Iteration": iteration,
                    "Pass": 1 if status == "Pass" else 0,
                    "Fail": 1 if status == "Fail" else 0
                })

                if status == "Pass":
                    total_pass +=1
                if status == "Fail":
                    total_fail +=1
        except FileNotFoundError:
            logging.error(f"File not found: {iter_txt_path}")

    if not rows:
        logging.debug("No data found to aggregate.")
        return

    df = pd.DataFrame(rows).sort_values(by="Iteration").reset_index(drop=True)
    output_csv_path = os.path.join(output_dir, f'{notebook_fname}_summary.csv')
    df.to_csv(output_csv_path, index=False)

    output_cnt_path = os.path.join(output_dir, f'{notebook_fname}_total_cnt.csv')
    df = pd.DataFrame({'Total_run': [total_pass+total_fail], 'Total_pass':[total_pass], 'Total_fail':[total_fail]})
    df.to_csv(output_cnt_path, index=False)

def main():
    # Example usage
    output_dir = "/home/yy2282/project/nb_test/testing-jupyter-notebook/outputs"
    notebook_path = "/home/yy2282/project/nb_test/testing-jupyter-notebook/outputs/predicting-alternate-dimension-transportation_chebyshev_0.95_1.ipynb"
    pytest_iterations = 1
    run_dir = "/home/yy2282/project/nb_test/testing-jupyter-notebook/examples/kaggle__abdelazizsami__predicting-alternate-dimension-transportation"


    # Run tests and aggregate results
    run_missing_tests(output_dir, notebook_path, pytest_iterations, run_dir)
    # aggregate_results_to_csv(output_dir,notebook_path)

if __name__ == "__main__":
    main()
