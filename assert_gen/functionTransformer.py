import ast
import os
import sys
import nbformat as nbf
import re

proj_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(proj_folder)
from utils.utils import setup_logger, preprocess_code

import ast

class FunctionTransformer(ast.NodeTransformer):
    def __init__(self, input_nb, logs_dir, output_nb, mapping):

        self.input_nb = input_nb
        self.output_nb = output_nb
        self.mapping = mapping

        self.assertion_list = []
        self.modified_lines = set()
        self.assertion_generated = {"Assertion_id": [], "Assertion": [], "Assertion_type": []}

        self.test_id = 0
        self.cell_no = 0
        self.logger = setup_logger(os.path.join(logs_dir, f'assertion_{self.__class__.__name__}.log'), self.__class__.__name__)

    def process_each_cell(self):
        ntbk = nbf.read(self.input_nb, nbf.NO_CONVERT)

        for cell_dict in self.mapping:

            for cell_no, assertions_config in cell_dict.items():
                self.logger.info(f"No. Cell: {cell_no}. Retriving: ntbk.cells[{cell_no-1}]")
                self.assertion_list = assertions_config
                cell = ntbk.cells[cell_no-1]

                self.cell_no = cell_no - 1

                code = cell.source
                self.logger.info(code)
                tree = ast.parse(code)
                transformed_tree = self.visit(tree)
                cell.source = ast.unparse(transformed_tree)

        nbf.write(ntbk, self.output_nb, version=4)

    def visit(self, node):
        prev_var = set()

        for idx, stmt in enumerate(node.body):

            for assertion in self.assertion_list:
                self.logger.info("Checking if this assertion should be added.")
                self.logger.info(assertion)

                if not self.is_assert_node(stmt):
                    if stmt.lineno == (assertion["lineno"]):

                        var_name = assertion["var"]

                        if var_name not in prev_var and var_name.startswith("nbtest_tmpvar"):
                            assign_node = self.create_assignment_node(var_name, stmt)
                            node.body[idx] = assign_node

                        pattern_dtype_listcomp = (
                            r"\[\s*str\s*\(\s*(?P<var>\w+)\s*\[\s*i\s*\]\s*\.s*dtype\s*\)"
                            r"\s*for\s+i\s+in\s+sorted\s*\(\s*(?P=var)\s*\.s*columns\s*\)\s*\]"
                        )
                        match_column_types = re.match(pattern_dtype_listcomp, assertion["args"][0])
                        if match_column_types:
                            assertion["func_name"] = "assert_column_types"
                            assertion["args"][0] = match_column_types.group("var")

                        match_column_names = re.match(r"sorted\s*\(\s*(?P<var>\w+)\s*\.s*columns\s*\)", assertion["args"][0])
                        if match_column_names:
                            assertion["func_name"] = "assert_column_names"
                            assertion["args"][0] = match_column_names.group("var")

                        pattern_nanmean_expr = (
                            r"np\.nanmean\s*\(\s*(?P<var>\w+)\.select_dtypes\s*\(\s*include\s*=\s*\[\s*['\"]number['\"]\s*\]\s*\)"
                            r"\.to_numpy\s*\(\s*\)\s*\)"
                        )
                        match_nanmean = re.match(pattern_nanmean_expr, assertion["args"][0])
                        if match_nanmean:
                            assertion["func_name"] = "assert_nanmean"
                            assertion["args"][0] = match_nanmean.group("var")

                        pattern_nanvar_expr = (
                            r"np\.nanvar\s*\(\s*(?P<var>\w+)\.select_dtypes\s*\(\s*include\s*=\s*\[\s*['\"]number['\"]\s*\]\s*\)"
                            r"\.to_numpy\s*\(\s*\)\s*\)"
                        )
                        match_nanvar = re.match(pattern_nanvar_expr, assertion["args"][0])
                        if match_nanvar:
                            assertion["func_name"] = "assert_nanvar"
                            assertion["args"][0] = match_nanvar.group("var")

                        self.logger.info(assertion["func_name"])

                        assert_node = self.create_assert_node(
                            stmt.lineno+1,
                            stmt.col_offset,
                            assertion["func_name"],
                            assertion["args"],
                            assertion["kwargs"],
                            f'{self.test_id}'
                        )

                        if assert_node is not None:
                            node.body.insert(idx + 1, assert_node)

                            prev_var.add(var_name)
                            args_part = ", ".join(str(arg) for arg in assertion["args"])

                            if assertion["kwargs"]:
                                kwargs_part = ", ".join([f"{key}={value}" for key, value in assertion["kwargs"].items()])
                                full_assert = f'nbtest.{assertion["func_name"]}({args_part}, {kwargs_part})'
                            else:
                                full_assert = f'nbtest.{assertion["func_name"]}({args_part})'

                            self.assertion_generated["Assertion_id"].append(f'{self.test_id}')
                            self.assertion_generated["Assertion"].append(full_assert)
                            self.assertion_generated["Assertion_type"].append(assertion["assert_type"])

                            self.test_id += 1

        return node

    def create_assignment_node(self, var_name, stmt):
        """
        Create an assignment node when the variable name starts with "nbtest_tmpvar".
        """
        return ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=stmt.value,
            lineno=stmt.lineno,
            col_offset=stmt.col_offset
        )

    def create_assert_node(self, lineno, col_offset, func_name, args, kwargs, test_id):
        """
        Create an assertion node dynamically based on provided function name and arguments.
        """
        args = args if args is not None else []
        kwargs = kwargs if kwargs is not None else {}


        ast_args = []
        if args:
            ast_args.append(ast.Name(id=args[0], ctx=ast.Load()))
            # ast_args.extend(self.to_ast_node(arg) for arg in args[1:])

            for arg in args[1:]:
                ast_arg = self.to_ast_node(arg)
                if ast_arg is None:
                    return None

                ast_args.append(ast_arg)


            ast_keywords = []
            for key, value in kwargs.items():
                ast_node = self.to_ast_node(value)
                if ast_node is None:
                    return None
                ast_keywords.append(ast.keyword(arg=key, value=ast_node))

            ast_keywords.append(ast.keyword(arg='test_id', value=self.to_ast_node(test_id)))

            return ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='nbtest', ctx=ast.Load()),
                        attr=func_name,
                        ctx=ast.Load()
                    ),
                    args=ast_args,
                    keywords=ast_keywords
                ),
                lineno=lineno,
                col_offset=col_offset
            )


    def to_ast_node(self, value):
        """Convert a value to an AST node."""
        if isinstance(value, ast.AST):
            return value
        elif isinstance(value, (int, float)):
            return ast.Constant(value=value)
        elif isinstance(value, str):
            return ast.Constant(value=value)
        elif isinstance(value, list):
            return ast.List(elts=[self.to_ast_node(v) for v in value], ctx=ast.Load())
        elif isinstance(value, tuple):
            return ast.Tuple(elts=[self.to_ast_node(v) for v in value], ctx=ast.Load())
        elif isinstance(value, dict):
            return ast.Dict(
                keys=[self.to_ast_node(k) for k in value.keys()],
                values=[self.to_ast_node(v) for v in value.values()]
            )
        elif value is None:
            return ast.Constant(value=None)
        else:
            self.logger.warning(f"Unsupported type of {value}: {type(value)}")
            return ast.Constant(value=None)

    def is_assert_node(self, stmt):
        """Check if stmt is an assertion node created by `create_assert_node`."""
        return (
            isinstance(stmt, ast.Expr) and
            isinstance(stmt.value, ast.Call) and
            isinstance(stmt.value.func, ast.Attribute) and
            isinstance(stmt.value.func.value, ast.Name) and
            stmt.value.func.value.id == "nbtest"  # Ensures it's calling nbtest.<func_name>
        )


def main():
    source_code = """
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
Sequential([Dense(x_train.shape[1], activation='relu'), Dense(24, activation='relu'), Dense(12, activation='relu'), Dense(10, activation='relu'), Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer=Adam(0.03), metrics=['Accuracy'])
history = model.fit(x_train, y_train, epochs=50, validation_batch_size=64, validation_data=(x_val, y_val))
    """

    cleaned_code = preprocess_code(source_code)


    assertion_list = [
        {
            "var": "nbtest_tmpvar_1",
            "lineno": 4,
            "func_name": "assert_allclose",
            "args": ["nbtest_tmpvar_1", "expected_value"],
            "kwargs": {"atol": 0.0003632869200276443},
            "assert_type":"model"
        },
        {
            "var": "model",
            "lineno": 5,
            "func_name": "assert_allclose",
            "args": ["model", "expected_value"],
            "kwargs": {"atol": 0.0003632869200276443},
            "assert_type":"model"
        },
        {
            "var": "test",
            "lineno": 4,
            "func_name": "assert_allclose",
            "args": ["test", "expected_value"],
            "kwargs": {"atol": 0.0003632869200276443},
            "assert_type":"model"
        }
    ]


    tree = ast.parse(cleaned_code)

    transformer = FunctionTransformer(assertion_list)
    modified_tree = transformer.visit(tree)

    modified_code = ast.unparse(modified_tree)
    print("\nModified Source Code:\n", modified_code)


if __name__ == "__main__":
    main()
