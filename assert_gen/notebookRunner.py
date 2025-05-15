import argparse
import os
import glob
import pickle
import subprocess
import sys
import json
from pathlib import Path

proj_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(proj_folder)
from utils.utils import setup_logger

config_path = os.path.join(proj_folder, "config.json")
global CONFIG_FILEPATH
with open(config_path, "r") as config_file:
    CONFIG_FILEPATH = json.load(config_file)["FILE_PATH"]

class NotebookRunner:
    def __init__(self, notebook_path, iterations, output_dir):
        self.notebook_path = os.path.abspath(notebook_path)
        self.iterations = iterations
        self.output_dir = os.path.abspath(output_dir)
        self.prev_ite = []
        self.notebook_fname = Path(self.notebook_path).stem.replace("_tmp", "").split(".ipynb")[0]

        self.instrumentation_pkl = os.path.join(output_dir, f"{self.notebook_fname}_{CONFIG_FILEPATH['INSTRUMENTATION_PKL']}")

        self.logs_dir = os.path.join(output_dir, f"{self.notebook_fname}_logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.logger = setup_logger(os.path.join(self.logs_dir, f'{self.__class__.__name__}.log'), self.__class__.__name__)

    def parse_previous_iterations(self):
        pattern = f"{self.notebook_fname}_instrumentation_*.pkl"
        for file in sorted(glob.glob(os.path.join(self.output_dir, pattern))):
            num = int(file.split("_")[-1].replace(".pkl", ""))
            self.prev_ite.append(num)

    def execute_notebook(self, input_path, output_path):
        try:
            result = subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    input_path,
                    "--output",
                    output_path
                ],
                capture_output=True,
                text=True,
                check=True,
                env={
                    **os.environ,
                    "COLLECT_VARS": "1",
                    "NBTEST_OUTPUT_DIR": self.output_dir
                }
            )
            self.logger.debug(f"Execution output: {result.stdout}")
            self.logger.debug(f"Execution output: {result.stderr}")
        except Exception as e:

            self.logger.error(f"Error when executing {self.notebook_path}", exc_info=True)

            self.logger.debug(f"Execution errors: {e.stderr}")

    def merge_pickle_files(self, output_file):
        ''' Read all pkl files with format: instrumentation_{num}.pkl
            Merge these pkl files into output_file (instrumentation.pkl)
            {num} is the pkl file for the ith iteration. {num} starts from 1
        '''
        self.logger.info("self.notebook_fname: {self.notebook_fname}")
        pickle_files = glob.glob(os.path.join(self.output_dir, f"{self.notebook_fname}_instrumentation_*.pkl"))
        if not pickle_files:
            self.logger.warning(f"No instrumentation files found: {self.notebook_fname}_instrumentation_*.pkl")
            return

        loaded_lists = []

        for file in pickle_files:
            with open(file, "rb") as f:
                loaded_lists.append(pickle.load(f))

        merged_list = [{} if isinstance(item, dict) else None for item in loaded_lists[0]]

        for lst in loaded_lists:
            for i, item in enumerate(lst):
                if (len(item) == 1 and next(iter(item.values()))["isNone"] is True) or merged_list[i] is None:
                    continue

                elif isinstance(item, dict):
                    for key, value in item.items():
                        if key not in merged_list[i]:
                            merged_list[i][key] = {}
                        for sub_key, sub_value in value.items():
                            if (sub_key == "cell_no" or sub_key == "line_no" or sub_key == "isNone"):
                                merged_list[i][key][sub_key] = sub_value
                            else:
                                if sub_key not in merged_list[i][key]:
                                    merged_list[i][key][sub_key] = []
                                merged_list[i][key][sub_key].append(sub_value)

        with open(output_file, "wb") as f:
            pickle.dump(merged_list, f)


    def run(self):
        self.parse_previous_iterations()

        for i in range(self.iterations):
            output_path = os.path.join(self.output_dir, f"{self.notebook_fname}_run_{i}.ipynb")
            if (i + 1) not in self.prev_ite:
                self.logger.debug(f"------------------ Iteration {i + 1} ---------------------------")
                self.execute_notebook(self.notebook_path, output_path)


        self.merge_pickle_files(self.instrumentation_pkl)



def main():
    parser = argparse.ArgumentParser(description="Run a notebook multiple times.")
    parser.add_argument("notebook", type=str, help="Path to the notebook file.")
    parser.add_argument("-n", type=int, default=1, help="Number of times to run the notebook.")
    parser.add_argument("-o", default=os.getcwd(), help="Output directory")


    args = parser.parse_args()

    runner = NotebookRunner(
        notebook_path=args.notebook,
        iterations=args.n,
        output_dir=args.o
    )
    runner.run()


if __name__ == "__main__":
    main()
