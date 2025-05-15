import os
import sys
import json
import argparse
import nbformat as nbf
import ast
import pandas as pd
from pathlib import Path

proj_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(proj_folder)

from properties.functionTransformer import FunctionTransformer
from utils.utils import setup_logger, preprocess_code, set_seed_random_state

config_path = os.path.join(proj_folder, "config.json")
with open(config_path, "r") as config_file:
    CONFIG_FILEPATH = json.load(config_file)["FILE_PATH"]

class PropertyFinder:
    def __init__(self, notebook, output_dir, original_nb_dir):
        self.notebook = notebook
        self.output_dir = output_dir
        self.notebook_fname = Path(notebook).stem
        self.original_nb_dir = original_nb_dir
        self.tmp_ipynb = os.path.join(self.original_nb_dir, f"{self.notebook_fname}_{CONFIG_FILEPATH['TMP_IPYNB']}")
        self.processed_ipynb = os.path.join(self.output_dir, f"{self.notebook_fname}_{CONFIG_FILEPATH['PROCESSED_IPYNB']}")
        self.properties_csv = os.path.join(output_dir, f"{self.notebook_fname}_{CONFIG_FILEPATH['PROPERTIES_CSV']}")
        self.properties_df = pd.DataFrame(columns=["Variable", "Assertion Type", "Code Cell", "Line Number"])

        # Setup logging
        logs_dir = os.path.join(output_dir, f"{self.notebook_fname}_logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.logger = setup_logger(os.path.join(logs_dir, f'{self.__class__.__name__}.log'), self.__class__.__name__)

    def find_property(self):
        """
        Analyze notebook cells and extract properties using FunctionTransformer.
        """
        try:
            ntbk = nbf.read(self.notebook, as_version=4)
        except:
            self.logger.error(f"File not exist: {self.notebook_fname}")
            return

        nb_dest = nbf.v4.new_notebook()
        # Insert import cell
        import_cell = nbf.v4.new_code_cell(source="import nbtest\nimport json\nimport numpy as np\nrandom_seed = np.random.randint(10000)")
        nb_dest.cells.append(import_cell)

        # Preprocess code
        for i, cell in enumerate(ntbk.cells):

            if cell.cell_type == "code":
                code = cell.source
                # Ignore HTML, JavaScript, or magic commands
                if not any(code.lstrip().startswith(prefix) for prefix in ["%%html", "%%javascript", "%%bash", "%%sh", "%%latex"]):
                    cleaned_code_no_seed = preprocess_code(cell.source)
                    cleaned_code = set_seed_random_state(cleaned_code_no_seed)
                    cleaned_cell= nbf.v4.new_code_cell(cleaned_code)
                    nb_dest.cells.append(cleaned_cell)
        nbf.write(nb_dest, self.processed_ipynb)

        function_transformer = FunctionTransformer(self.processed_ipynb, self.output_dir, self.tmp_ipynb)
        function_transformer.process_each_cell()

        for property_dict in function_transformer.found_properties:

            for var_name, info in property_dict.items():
                temp_df = pd.DataFrame({
                    "Variable": [info["assert_var"]],
                    "Assertion Type": [info["assert_type"]],
                    "Code Cell": [info["cell_no"]],
                    "Line Number": [info["line"]],
                })
                self.properties_df = pd.concat([self.properties_df, temp_df], ignore_index=True)

        # nbf.write(ntbk, self.tmp_ipynb)
        self.properties_df.to_csv(self.properties_csv, index=False)
        self.logger.info(f"Properties saved to {self.properties_csv}")

    def run(self):
        """
        Run the entire property finding and notebook updating process.
        """
        self.find_property()

def main():
    parser = argparse.ArgumentParser(description="Generate properties for a Jupyter notebook.")
    parser.add_argument("notebook", type=str, help="Path to the Jupyter notebook file")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--original_dir", help="Directory to original folders with notebook")
    args = parser.parse_args()

    notebook_path = os.path.abspath(args.notebook)
    output_dir = os.path.abspath(args.output_dir)
    original_dir = os.path.abspath(args.original_dir)

    property_finder = PropertyFinder(notebook_path, output_dir, original_dir)
    property_finder.run()

if __name__ == "__main__":
    main()
