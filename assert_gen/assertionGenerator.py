import pandas as pd
import nbformat as nbf
import os
import numpy as np
import sys
import pickle
import traceback
import ast
from collections import defaultdict
import json
import collections.abc

proj_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(proj_folder)
from utils.utils import setup_logger, preprocess_code
from assert_gen.functionTransformer import FunctionTransformer

config_path = os.path.join(proj_folder, "config.json")
with open(config_path, "r") as config_file:
    CONFIG_FILEPATH = json.load(config_file)["FILE_PATH"]

class AssertionGenerator:
    def __init__(self, instrumentation_pkl, original_copy_ipynb,properties_csv, output_dir, notebook_fname, stats_method, conf_level, iteration):
        """
        Initialize the AssertionGenerator with required file paths and parameters.
        """
        self.instrumentation_pkl = instrumentation_pkl
        self.original_copy_ipynb = original_copy_ipynb
        self.properties_csv = properties_csv
        self.output_dir = output_dir
        self.notebook_fname = notebook_fname
        self.stats_method = stats_method
        self.conf_level = conf_level
        self.iteration = iteration
        self.processed_ipynb = os.path.join(self.output_dir, f"{self.notebook_fname}_{CONFIG_FILEPATH['PROCESSED_IPYNB']}")

        self.mapping = {}

        self.logs_dir = os.path.join(output_dir, f"{self.notebook_fname}_logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.logger = setup_logger(os.path.join(self.logs_dir, f'{self.__class__.__name__}.log'), self.__class__.__name__)

        try:
            self.instrument_value = pd.read_pickle(self.instrumentation_pkl)
        except Exception as e:
            self.logger.error(f"Failed to read pickle file {self.instrumentation_pkl}: {e}")
            raise

        try:
            self.properties_df = pd.read_csv(self.properties_csv)
        except Exception as e:
            self.logger.error(f"Failed to read CSV file {self.properties_csv}: {e}")
            raise

        try:
            self.ntbk = nbf.read(self.original_copy_ipynb, nbf.NO_CONVERT)
        except Exception as e:
            self.logger.error(f"Failed to read notebook file {self.original_copy_ipynb}: {e}")
            raise

    def calculate_stats_bound(self, values, method, confidence_level=0.95):
        """
        Calculate the relative tolerance and mean value based on statistical methods.
        """
        try:

            values = [float(x) for x in values]

            mean_val = np.mean(values)
            std_dev = np.std(values, ddof=0)

            if method == "chebyshev":
                epsilon = np.sqrt(1 / (1 - confidence_level)) * std_dev
            elif method == "hoeffding":
                epsilon = np.sqrt(np.log(2 / (1 - confidence_level)) / (2 * len(values)))
        except Exception as e:
            self.logger.error(f"Error in calcualte bound for {values}" + traceback.format_exc())
            return None

        return mean_val, epsilon


    def get_assert_type(self, value):

        assert_type = "assert_equal"

        if all(isinstance(item, (int, float, np.number)) for item in value):
            assert_type = "assert_allclose"

        if all(isinstance(item, bool) for item in value):
            assert_type = f"assert_{str(value[0]).lower()}"

        if not isinstance(value, list) or len(value) == 0:
            return "unsupported"

        if isinstance(value, list) or isinstance(value, pd.Series):
            first_value = value.iloc[0] if isinstance(value, pd.Series) else value[0]
            if first_value is None:
                return "unsupported"

        if all(isinstance(item, dict) for item in value):
            if isinstance(value, (pd.Series, np.ndarray)):
                value = value.tolist()
            first_dict = value[0]
            if not all(d == first_dict for d in value):
                return "unsupported"

        if isinstance(value[0], (list, tuple)):
            if len(value[0]) == 0:
                return "unsupported"

            if not isinstance(value[0][0], (int, float, str, bool)):
                return "unsupported"

        return assert_type


    def decide_assert_type_bound(self, assertion_config):
        instrument_dict = assertion_config["val"]
        assert_var = assertion_config["var"]
        assertion_config["func_name"] = None
        assertion_config["args"] = []
        assertion_config["kwargs"] = {}

        if instrument_dict:
            for feature, value in instrument_dict.items():
                assert_type = self.get_assert_type(value)
                updated_feature = feature.replace("assert_var", assert_var)

                if assert_type == "assert_allclose":
                    results = self.calculate_stats_bound(
                                value,
                                self.stats_method,
                                self.conf_level
                            )
                    if results is not None:
                        assert_values, assert_rtol = results
                        assertion_config["func_name"] = assert_type
                        assertion_config["args"] = [str(updated_feature), assert_values]
                        assertion_config["kwargs"] = {"atol": assert_rtol}
                        continue
                elif assert_type == "unsupported":
                    continue
                else:
                    first_item = value[0]
                    has_same_item = all((first_item == item) for item in value[1:])

                    if has_same_item:
                        assertion_config["func_name"] = assert_type
                        assertion_config["args"] = [str(updated_feature), value[0]]
                        assertion_config["kwargs"] = {}
                    else:
                        continue
        return assertion_config

    def generate_assertions(self):

        assertion_generated = {"Assertion_id": [], "Assertion": [], "Assertion_type": []}

        ntbk = nbf.read(self.original_copy_ipynb, as_version=4)

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
                    cleaned_cell= nbf.v4.new_code_cell(cleaned_code_no_seed)
                    nb_dest.cells.append(cleaned_cell)
        nbf.write(nb_dest, self.processed_ipynb)


        for cell_dict in self.mapping:

            for cell_no, assertions_config in cell_dict.items():
                for assertion_config in assertions_config:
                    self.decide_assert_type_bound(assertion_config)

                assertions_config[:] = [ac for ac in assertions_config if ac["func_name"] is not None]

        notebook_w_assert = os.path.join(self.output_dir, f"{self.notebook_fname}_{self.stats_method}_{self.conf_level}_{self.iteration}.ipynb")

        function_transformer = FunctionTransformer(self.processed_ipynb, self.logs_dir, notebook_w_assert, self.mapping)
        function_transformer.process_each_cell()

        assertion_generated = {key: assertion_generated[key] + function_transformer.assertion_generated[key] for key in assertion_generated}
        df = pd.DataFrame(assertion_generated)
        df.to_csv(os.path.join(self.output_dir, f"{self.notebook_fname}_assertions.csv"), index=False)

    def map_var_to_property(self):
        self.logger.info("Creating mapping between the property and the instrumentated value.")
        properties_df = pd.read_csv(self.properties_csv)

        with open(self.instrumentation_pkl, "rb") as f:
            pickle_data = pickle.load(f)

        cell_mapping = defaultdict(list)

        self.logger.info(f"Checking the contents in {self.instrumentation_pkl}")
        for pickle_row in pickle_data:
            self.logger.info(pickle_row)
            self.logger.info('\n')

        for _, row in properties_df.iterrows():
            variable = row["Variable"]
            assertion_type = row["Assertion Type"]
            code_cell = row["Code Cell"]
            line_number = row["Line Number"]
            is_assertion = False

            for pickle_row in pickle_data:
                if pickle_row:
                    variable_name = next(iter(pickle_row.keys()))
                    instrument_info = next(iter(pickle_row.values()))

                    if variable_name == variable:

                        if instrument_info["cell_no"] == code_cell and instrument_info["line_no"] == line_number:
                            values = {key: value for key, value in instrument_info.items() if key not in {"cell_no", "line_no", "isNone"}}

                            for feature, value in values.items():

                                assertion_entry = {
                                    "var": variable,
                                    "assert_type": assertion_type,
                                    "lineno": line_number,
                                    "val": {feature: value}
                                }


                                cell_mapping[code_cell].append(assertion_entry)
                                is_assertion = True
            if is_assertion:
                self.logger.info(f"{variable} at Cell {code_cell} Line {line_number} has found the instrumentation value")
            else:
                self.logger.info(f"{variable} at Cell {code_cell} Line {line_number} didn't find the instrumentation value. No matching code_cell and line_number or variable name.")


        self.mapping = [{cell_no: assertions} for cell_no, assertions in cell_mapping.items()]
        self.logger.info("Checking self.mapping")
        for mapping in self.mapping:
            self.logger.info(mapping)

    def run(self):
        if os.path.isfile(self.instrumentation_pkl):
            self.map_var_to_property()
            self.generate_assertions()
        else:
            self.logger.warning(f"No assertions generated because {self.instrumentation_pkl} is not found. ")


# Example Usage
if __name__ == "__main__":
    instrumentation_pkl = "/home/yy2282/project/nb_test/testing-jupyter-notebook/outputs/house-prices-advanced-regression-techniques_instrumentation.pkl"
    original_copy_ipynb = "/home/yy2282/project/nb_test/testing-jupyter-notebook/outputs/house-prices-advanced-regression-techniques.ipynb"
    properties_csv = "/home/yy2282/project/nb_test/testing-jupyter-notebook/outputs/house-prices-advanced-regression-techniques_properties.csv"
    output_dir = "/home/yy2282/project/nb_test/testing-jupyter-notebook/outputs"
    notebook_fname = "house-prices-advanced-regression-techniques"
    stats_method = "chebyshev"
    conf_level = 0.95
    iteration = 2

    processor = AssertionGenerator(
        instrumentation_pkl=instrumentation_pkl,
        original_copy_ipynb = original_copy_ipynb,
        properties_csv=properties_csv,
        output_dir=output_dir,
        notebook_fname=notebook_fname,
        stats_method=stats_method,
        conf_level = conf_level,
        iteration = iteration
    )

    processor.run()
