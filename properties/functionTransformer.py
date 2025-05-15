import ast
import os
import json
import sys
from collections import defaultdict
from pathlib import Path
import nbformat as nbf
import pandas as pd
import argparse

proj_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(proj_folder)
global config_path
config_path = os.path.join(proj_folder, "config.json")

from utils.utils import preprocess_code
from utils.utils import setup_logger


class FunctionTransformer(ast.NodeTransformer):

    def __init__(self, notebook_path, output_dir, output_notebook):
        self.output_file = output_notebook
        self.found_properties = []
        self.cell_no = 0
        self.notebook_path = notebook_path
        self.notebook_fname = Path(output_notebook).stem.replace("_tmp", "").split(".ipynb")[0]
        self.def_use_map = defaultdict(list)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.def_set = set()
        self.need_instrument = False
        self.type_keywords = {
                "DATASET": ["ss", "ts", "ds", "train", "test_data", "dataset", "data", "correlation", "x_test", "df", "X_test", "y_val",
                             "X_train", "y_train", "X_val", "y_val"],
                "MODEL_PER": ["accuracy", "loss", "precision", "recall", "f1", "auc", "correct", "incorrect", "results", "error", "y_pred",
                              "acc"],
                "MODEL_ARCH": ["model", "rfr"]
            }


        with open(config_path, "r") as config_file:
            self.known_apis = json.load(config_file)["KNOWN_APIS"]

        logs_dir = os.path.join(output_dir, f"{self.notebook_fname}_logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.logger = setup_logger(os.path.join(logs_dir, f'{self.__class__.__name__}.log'), self.__class__.__name__)
        self.logger_no_instrument = setup_logger(os.path.join(logs_dir, 'duplicate_assertions.log'), 'duplicate_assertions')
        self.tmpvar_count = 0

    def process_each_cell(self):
        ntbk = nbf.read(self.notebook_path, nbf.NO_CONVERT)

        for i, cell in enumerate(ntbk.cells):

            if cell.cell_type == "code":
                try:
                    code = cell.source
                    self.cell_no = i+1

                    tree = ast.parse(code)

                    transformed_tree = self.visit(tree)
                    cell.source = ast.unparse(transformed_tree)
                except Exception as e:
                    self.logger.error(f"Error processing cell {i+1}: {e}")

        nbf.write(ntbk, self.output_file)

    def add_property_dict(self, var_name, assert_type, assert_var, func_name, line):
        """
        Add a property dictionary to the list of found properties.
        """
        self.logger.info(f"Checking if ({var_name}, {self.cell_no},{line}) should be added.")
        self.logger.info("Current self.def_set: ")
        self.logger.info(self.def_set)

        def_key = (var_name, self.cell_no, line)

        def_value = f"{var_name}_{self.def_use_map.get(def_key)}"

        self.logger.info(f"The def_value is:")
        self.logger.info(def_value)
        if def_value not in self.def_set:
            self.logger.info(f"({var_name}, {self.cell_no},{line}) is not in the self.def_set. Instrumenting")
            # Only instrument the value with different def
            if "None" not in def_value:
                self.def_set.add(def_value)
            self.need_instrument = True

            property_dict = {
                var_name: {
                    'assert_type': f"{assert_type}__{func_name}",
                    'assert_var': assert_var,
                    'func_name': func_name,
                    'line': line,
                    'cell_no':self.cell_no
                }
            }
            self.found_properties.append(property_dict)
        else:
            self.need_instrument = False
            self.logger.info(f"({var_name}, {self.cell_no},{line}) is already in the self.def_set. Stop instrumenting")
            self.logger_no_instrument.info(f"({var_name}, {self.cell_no},{line})")

    def get_func_name(self, func):

        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return func.attr
        return None

    def visit_Assign(self, node):
        """
        Process assignment statements to identify relevant metrics.
        """

        if not isinstance(node.targets[0], (ast.Name, ast.Tuple)) or not isinstance(node.value, ast.Call):
            return node

        func_name = self.get_func_name(node.value.func)
        if func_name not in self.known_apis:
            return node
        if func_name == "classification_report":
            new_nodes = [node]
            assert_type = self.known_apis[func_name]["assert_type"]
            self.logger.info("classification_report")
            arg = node.value
            report_call = ast.Call(
                            func=ast.Name(id='classification_report', ctx = ast.Load()),
                            args = [ast.copy_location(a, arg) for a in arg.args],
                            keywords=[ast.copy_location(ast.keyword(
                                                arg=kw.arg,
                                                value = ast.copy_location(kw.value, kw.value)
                                                ), kw) for kw in arg.keywords]
                        )

            has_output_dict = False
            for keyword in report_call.keywords:
                if keyword.arg == 'output_dict':
                    has_output_dict = True
                    break

            if not has_output_dict:
                report_call.keywords.append(ast.keyword(arg='output_dict', value = ast.Constant(value=True)))

            value = ast.unparse(report_call)

            metric_list = [
                ("accuracy",),
                ("macro avg", "precision"),
                ("macro avg", "recall"),
                ("macro avg", "f1-score"),
                ("macro avg", "support"),
                ("weighted avg", "precision"),
                ("weighted avg", "recall"),
                ("weighted avg", "f1-score"),
                ("weighted avg", "support"),
            ]
            for metric_keys in metric_list:
                metric = ''.join(f'["{key}"]' for key in metric_keys)
                instrument_value =value+metric
                self.add_property_dict(instrument_value, assert_type, instrument_value, func_name, node.lineno)

                if self.need_instrument and assert_type != "UNKNOWN":
                    assert_node = self.create_instrument_node(instrument_value, self.cell_no, node.lineno, api=func_name)
                    insert_index = self.get_insert_position(node)
                    new_nodes.insert(insert_index, assert_node)
        else:

            targets = node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else [node.targets[0]]
            new_nodes = [node]

            for target in targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    assert_type = self.known_apis[func_name]["assert_type"]
                    assert_var = var_name

                    self.add_property_dict(var_name, assert_type, assert_var, func_name, node.lineno)

                    if self.need_instrument and assert_type != "UNKNOWN":
                        assert_node = self.create_instrument_node(var_name, self.cell_no, node.lineno, api=func_name)
                        insert_index = self.get_insert_position(node)
                        new_nodes.insert(insert_index, assert_node)

        return new_nodes


    def visit_Expr(self, node):
        """
            Process expressions for implicit prints or outputs.
            e.g.,
            train_df.head()
            plt.show()
            "Test string".format(var)
            print(test_values.shape)
            print(X_test.shape)
        """
        new_nodes = [node]
        assert_type = "UNKNOWN"
        if isinstance(node.value, ast.Name):
            var_name = self.get_base_id(node.value)
            stored_value = var_name.lower()
            assert_type = self.determine_type(stored_value)

            self.add_property_dict(
                var_name=var_name,
                assert_type=assert_type,
                assert_var=var_name,
                func_name="SINGLE_VAR",
                line=node.lineno
            )

            if self.need_instrument and assert_type != "UNKNOWN":
                assert_node = self.create_instrument_node(var_name, self.cell_no, node.lineno)
                insert_index = self.get_insert_position(node)
                new_nodes.insert(insert_index, assert_node)
            return new_nodes

        # Handle attribute access (e.g., `train_data.shape`)
        if isinstance(node.value, ast.Attribute):
            var_name = ast.unparse(node.value)
            assert_type = self.determine_type(node.value)

            self.add_property_dict(
                var_name=var_name,
                assert_type=assert_type,
                assert_var=var_name,
                func_name="ATTRIBUTE_ACCESS",
                line=node.lineno
            )

            if self.need_instrument and assert_type != "UNKNOWN":
                assert_node = self.create_instrument_node(var_name, self.cell_no, node.lineno)
                new_nodes.append(assert_node)
            return new_nodes

        if isinstance(node.value, ast.Call):
            func_name = self.get_func_name(node.value.func)

            # Handle `print` function calls
            if func_name == "print":
                key = None
                assert_type = "PRINT"
                if node.value.args:
                    # If the first argument is a string literal, use it as the key
                    if isinstance(node.value.args[0], ast.Constant) and isinstance(node.value.args[0].value, str):
                        stored_value = self.get_base_id(node.value.args[0]).lower()
                        assert_type = self.determine_type(stored_value)
                        key = repr(node.value.args[0].value)

                    for arg in node.value.args:
                        # Handle nested function calls or variables
                        if isinstance(arg, ast.Call):
                            if isinstance(arg.func, ast.Attribute) and arg.func.attr == "format":
                                for format_arg in arg.args:  # Extract only arguments to .format()
                                    # Check for variable name
                                    assert_type = self.determine_type(node.value)

                                    extracted_value = ast.unparse(format_arg)
                                    self.add_property_dict(key or extracted_value, assert_type, extracted_value, "print", node.lineno)

                                    if self.need_instrument and assert_type != "UNKNOWN":
                                        assert_node = self.create_instrument_node(extracted_value, self.cell_no, node.lineno)
                                        insert_index = self.get_insert_position(node)
                                        new_nodes.insert(insert_index, assert_node)
                            else:
                                value = ast.unparse(arg)

                                # # Check for variable name
                                assert_type = self.determine_type(node.value)

                                func_name = self.get_func_name(arg.func)
                                if func_name in self.known_apis:
                                    assert_type = self.known_apis[func_name]["assert_type"]
                                else:
                                    func_name = "print_call"

                                if func_name == "classification_report":
                                    report_call = ast.Call(
                                                    func=ast.Name(id='classification_report', ctx = ast.Load()),
                                                    args = [ast.copy_location(a, arg) for a in arg.args],
                                                    keywords=[ast.copy_location(ast.keyword(
                                                                        arg=kw.arg,
                                                                        value = ast.copy_location(kw.value, kw.value)
                                                                        ), kw) for kw in arg.keywords]
                                                )

                                    has_output_dict = False
                                    for keyword in report_call.keywords:
                                        if keyword.arg == 'output_dict':
                                            has_output_dict = True
                                            break

                                    if not has_output_dict:
                                        report_call.keywords.append(ast.keyword(arg='output_dict', value = ast.Constant(value=True)))

                                    value = ast.unparse(report_call)

                                    metric_list = [
                                        ("accuracy",),
                                        ("macro avg", "precision"),
                                        ("macro avg", "recall"),
                                        ("macro avg", "f1-score"),
                                        ("macro avg", "support"),
                                        ("weighted avg", "precision"),
                                        ("weighted avg", "recall"),
                                        ("weighted avg", "f1-score"),
                                        ("weighted avg", "support"),
                                    ]
                                    for metric_keys in metric_list:
                                        metric = ''.join(f'["{key}"]' for key in metric_keys)
                                        instrument_value =value+metric
                                        self.add_property_dict(instrument_value, assert_type, instrument_value, func_name, node.lineno)

                                        if self.need_instrument and assert_type != "UNKNOWN":
                                            assert_node = self.create_instrument_node(instrument_value, self.cell_no, node.lineno, api=func_name)
                                            insert_index = self.get_insert_position(node)
                                            new_nodes.insert(insert_index, assert_node)

                                else:
                                    self.add_property_dict(key or value, assert_type, value, func_name, node.lineno)

                                    if self.need_instrument and assert_type != "UNKNOWN":
                                        assert_node = self.create_instrument_node(value, self.cell_no, node.lineno,api=func_name)
                                        insert_index = self.get_insert_position(node)
                                        new_nodes.insert(insert_index, assert_node)

                        elif isinstance(arg, ast.BinOp): #print('Train Accuracy:', l1.score(x_train, y_train) * 100)
                            if isinstance(arg.left, ast.Constant):
                                stored_value = self.get_base_id(arg.left).lower()
                                assert_type = self.determine_type(stored_value)

                            if isinstance(arg.right, ast.Call):
                                var_name = ast.unparse(arg.right)

                                self.add_property_dict(
                                    var_name=var_name,
                                    assert_type=assert_type,
                                    assert_var=var_name,
                                    func_name=func_name,
                                    line=node.lineno
                                )
                                if self.need_instrument and assert_type != "UNKNOWN":
                                    assert_node = self.create_instrument_node(var_name, self.cell_no, node.lineno)
                                    insert_index = self.get_insert_position(node)
                                    new_nodes.insert(insert_index, assert_node)

                            elif isinstance(arg.right, ast.Tuple):
                                for var in arg.right.elts:
                                    var_name = ast.unparse(var)

                                    self.add_property_dict(
                                        var_name=var_name,
                                        assert_type=assert_type,
                                        assert_var=var_name,
                                        func_name=func_name,
                                        line=node.lineno
                                    )
                                    if self.need_instrument and assert_type != "UNKNOWN":
                                        assert_node = self.create_instrument_node(var_name, self.cell_no, node.lineno)
                                        insert_index = self.get_insert_position(node)
                                        new_nodes.insert(insert_index, assert_node)


                        elif isinstance(arg, ast.Attribute):  # Handles cases like `test_values.shape`
                            var_name = ast.unparse(arg)
                            self.add_property_dict(
                                var_name=var_name,
                                assert_type=assert_type,
                                assert_var=var_name,
                                func_name=func_name,
                                line=node.lineno
                            )

                            if self.need_instrument and assert_type != "UNKNOWN":
                                assert_node = self.create_instrument_node(var_name, self.cell_no, node.lineno)
                                insert_index = self.get_insert_position(node)
                                new_nodes.insert(insert_index, assert_node)

                        elif isinstance(arg, ast.Name):  # Handle variables like `test_values`
                            var_name = arg.id
                            self.add_property_dict(
                                var_name=var_name,
                                assert_type=assert_type,
                                assert_var=var_name,
                                func_name="print",
                                line=node.lineno
                            )

                            if self.need_instrument and assert_type != "UNKNOWN":
                                assert_node = self.create_instrument_node(var_name, self.cell_no, node.lineno)
                                insert_index = self.get_insert_position(node)
                                new_nodes.insert(insert_index, assert_node)

                        # Handle f-strings
                        elif isinstance(arg, ast.JoinedStr):

                            for value in arg.values:
                                if isinstance(value, ast.Constant):
                                    stored_value = self.get_base_id(value).lower()
                                    assert_type = self.determine_type(stored_value)

                                if isinstance(value, ast.FormattedValue):
                                    extracted_value = ast.unparse(value.value)
                                    self.add_property_dict(key or extracted_value,assert_type, extracted_value, "print", node.lineno)

                                    if self.need_instrument and assert_type != "UNKNOWN":
                                        assert_node = self.create_instrument_node(extracted_value, self.cell_no, node.lineno)
                                        insert_index = self.get_insert_position(node)
                                        new_nodes.insert(insert_index, assert_node)

            # General case: Handle non-`print` function calls
            else:
                if func_name in self.known_apis:
                    assert_type = self.known_apis[func_name]["assert_type"]
                else:
                    assert_type = self.determine_type(node.value)

                # Handle .format() calls
                if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "format":
                    for arg in node.value.args:
                        if isinstance(arg, ast.Name):
                            extracted_value = arg.id
                        else:
                            extracted_value = ast.unparse(arg)

                        self.add_property_dict(
                            var_name=ast.unparse(node.value),
                            assert_type=assert_type,
                            assert_var=extracted_value,
                            func_name=node.value.func.attr,
                            line=node.lineno
                        )

                        if self.need_instrument and assert_type != "UNKNOWN":
                            assert_node = self.create_instrument_node(extracted_value, self.cell_no, node.lineno)
                            insert_index = self.get_insert_position(node)
                            new_nodes.insert(insert_index, assert_node)
                else:
                    # General function call handling
                    temp_var_name = f"nbtest_tmpvar_{self.tmpvar_count}"

                    # Create an assignment node for the temporary variable
                    assign_node = ast.Assign(
                        targets=[ast.Name(id=temp_var_name, ctx=ast.Store())],
                        value=node.value
                    )
                    ast.fix_missing_locations(assign_node)

                    # Add property and assertion for the temporary variable
                    self.add_property_dict(
                        var_name=temp_var_name,
                        assert_type=assert_type,
                        assert_var=temp_var_name,
                        func_name=func_name,
                        line=node.lineno
                    )

                    if self.need_instrument and assert_type != "UNKNOWN":
                        assert_node = self.create_instrument_node(temp_var_name, self.cell_no, node.lineno, api=func_name)
                        self.tmpvar_count += 1
                        # Replace the original node with the assign_node and append the assert_node
                        parent = getattr(node, "parent", None)
                        if parent and hasattr(parent, "body"):
                            body = parent.body
                            index = body.index(node)
                            body[index] = assign_node
                            body.insert(index + 1, assert_node)

                        return [assign_node, assert_node]

        return new_nodes

    def create_instrument_node(self, var_name, cell_no, line_no, api=None):

        instrument_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='nbtest', ctx=ast.Load()),
                    attr='instrument',
                    ctx=ast.Load()
                ),
                args=[
                    ast.Name(id=var_name, ctx=ast.Load()),
                    ast.Constant(value=cell_no, kind=None),
                    ast.Constant(value=line_no, kind=None),
                    ast.Constant(value=self.notebook_fname, kind=None),
                    ast.Constant(value=var_name, kind=None)
                ],
                keywords=[ast.keyword(arg='api', value=ast.Constant(value=api))] if api else []
            )
        )
        ast.fix_missing_locations(instrument_call)
        return instrument_call

    def get_insert_position(self, node):
        """
        Determine the position to insert the assert_node based on the current and next line.
        """
        parent = getattr(node, "parent", None)
        if not parent or not hasattr(parent, "body"):
            return 1

        body = parent.body
        current_index = body.index(node)

        # If this is the last node, append at the end
        if current_index == len(body) - 1:
            return len(body)

        # Otherwise, insert before the next node
        return current_index + 1

    def get_base_id(self,node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self.get_base_id(node.func)
        elif isinstance(node, ast.Attribute):
            return self.get_base_id(node.value)
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            return None

    def determine_type(self, node_value):

        stored_value = self.get_base_id(node_value)
        if stored_value is not None:
            stored_value = stored_value.lower()
            lowered = stored_value.lower()
            for type_name, keywords in self.type_keywords.items():
                if any(keyword in lowered for keyword in keywords):
                    return type_name
        return "UNKNOWN"

def main():

    code = """
print(classification_report(y_val, y_pred))
"""
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=str, help="notebook to parse")
    parser.add_argument('--output_dir', type=str, help="output folder")
    parser.add_argument('--output_notebook', type =str, help="output notebook path")

    args = parser.parse_args()

    function_transformer = FunctionTransformer(args.notebook, args.output_dir, args.output_notebook)
    function_transformer.process_each_cell()

    for row in function_transformer.found_properties:
        print(row)

if __name__ == "__main__":
    main()
