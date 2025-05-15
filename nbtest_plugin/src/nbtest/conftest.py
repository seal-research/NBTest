from __future__ import print_function

# import the pytest API
import pytest
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path
import ast
from queue import Empty
import os
import sys

# for reading notebook files
import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_output

# Kernel for running notebooks
from .kernel import RunningKernel, CURRENT_ENV_KERNEL_NAME

err_tracebacks = []

# define colours for pretty outputs
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

class nocolors:
    HEADER = ''
    OKBLUE = ''
    OKGREEN = ''
    WARNING = ''
    FAIL = ''
    ENDC = ''

def convert_to_camelcase(assertion_name):
    components = assertion_name.split('_')
    return components[0] + ''.join(x.capitalize() for x in components[1:])

class ReplaceNbtest(ast.NodeTransformer):
    def __init__(self, test_node):
        self.test_node = test_node
    def visit_Call(self, node):
        keywords = [kw for kw in node.keywords if kw.arg != 'test_id']

        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'nbtest'
            and ast.unparse(node) == ast.unparse(self.test_node)):

                if node.func.attr in ['assert_equal', 'assert_allclose']:
                    new_node = ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id='np', ctx=ast.Load()),
                                attr='testing',
                                ctx=ast.Load(),
                            ),
                            attr=node.func.attr,
                            ctx=ast.Load()
                        ),
                        args=node.args,
                        keywords=keywords
                    )
                    return ast.copy_location(new_node, node)

                elif node.func.attr == "assert_column_names":
                    new_node = ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="testing", ctx=ast.Load()),
                            attr="assert_equal",
                            ctx=ast.Load()
                        ),
                        args=[
                            ast.Call(
                                func=ast.Name(id="sorted", ctx=ast.Load()),
                                args=[ast.Attribute(value=node.args[0], attr="columns", ctx=ast.Load())],
                                keywords=[]
                            ),
                            node.args[1]
                        ],
                        keywords=[]
                    )
                    return ast.copy_location(new_node, node)

                elif node.func.attr == "assert_column_types":
                    df_arg_node = node.args[0]

                    list_comp_node = ast.ListComp(
                        # Element: str(df[i].dtype)
                        elt=ast.Call(
                            func=ast.Name(id='str', ctx=ast.Load()),
                            args=[
                                ast.Attribute(
                                    value=ast.Subscript(
                                        value=df_arg_node, # Use the extracted df node
                                        slice=ast.Index(value=ast.Name(id='i', ctx=ast.Load())) if sys.version_info < (3,9) else ast.Name(id='i', ctx=ast.Load()), # Use 'i' as index
                                        ctx=ast.Load()
                                    ),
                                    attr='dtype',
                                    ctx=ast.Load()
                                )
                            ],
                            keywords=[]
                        ),
                        # Generators: for i in sorted(df.columns)
                        generators=[
                            ast.comprehension(
                                target=ast.Name(id='i', ctx=ast.Store()), # Loop variable 'i'
                                iter=ast.Call(
                                    func=ast.Name(id='sorted', ctx=ast.Load()),
                                    args=[
                                        ast.Attribute(
                                            value=df_arg_node, # Use the extracted df node
                                            attr='columns',
                                            ctx=ast.Load()
                                        )
                                    ],
                                    keywords=[]
                                ),
                                ifs=[],
                                is_async=0
                            )
                        ]
                    )

                    new_node = ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id="np", ctx=ast.Load()),
                                attr="testing",
                                ctx=ast.Load()
                            ),
                            attr="assert_equal",
                            ctx=ast.Load()
                        ),
                        args=[
                            list_comp_node,
                            node.args[1]
                        ],
                        keywords=[]
                    )

                    return ast.copy_location(new_node, node)

                elif node.func.attr == 'assert_nanmean':
                    df_node = node.args[0]

                    # a.select_dtypes(include=['number']).to_numpy()
                    select_dtypes_node = ast.Call(
                        func = ast.Attribute(
                            value = ast.Call(
                                func = ast.Attribute(
                                    value = df_node,
                                    attr='select_dtypes',
                                    ctx = ast.Load()),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg='include',
                                        value=ast.List(
                                            elts=[
                                                ast.Constant(value='number')],
                                            ctx=ast.Load()))]),
                            attr='to_numpy',
                            ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    )

                    # np.testing.assert_allclose(np.nanmean(a.select_dtypes(include=['number']).to_numpy()), b, rtol=rtol, atol=atol, err_msg=err_msg)
                    new_node = ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id='np', ctx=ast.Load()),
                                attr='testing',
                                ctx=ast.Load()),
                            attr='assert_allclose',
                            ctx=ast.Load()),
                        args=[
                            ast.Call(
                                ast.Attribute(
                                    value=ast.Name(id='np', ctx=ast.Load()),
                                    attr='nanmean',
                                    ctx=ast.Load()),
                                args = [select_dtypes_node],
                                keywords = []
                            ),
                            node.args[1]
                        ],
                        keywords = keywords
                    )

                    return ast.copy_location(new_node, node)

                elif node.func.attr == 'assert_nanvar':
                    df_node = node.args[0]

                    # a.select_dtypes(include=['number']).to_numpy()
                    select_dtypes_node = ast.Call(
                        func = ast.Attribute(
                            value = ast.Call(
                                func = ast.Attribute(
                                    value = df_node,
                                    attr='select_dtypes',
                                    ctx = ast.Load()),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg='include',
                                        value=ast.List(
                                            elts=[
                                                ast.Constant(value='number')],
                                            ctx=ast.Load()))]),
                            attr='to_numpy',
                            ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    )

                    # np.testing.assert_allclose(np.nanvar(a.select_dtypes(include=['number']).to_numpy()), b, rtol=rtol, atol=atol, err_msg=err_msg)
                    new_node = ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id='np', ctx=ast.Load()),
                                attr='testing',
                                ctx=ast.Load()),
                            attr='assert_allclose',
                            ctx=ast.Load()),
                        args=[
                            ast.Call(
                                ast.Attribute(
                                    value=ast.Name(id='np', ctx=ast.Load()),
                                    attr='nanvar',
                                    ctx=ast.Load()),
                                args = [select_dtypes_node],
                                keywords = []
                            ),
                            node.args[1]
                        ],
                        keywords = keywords
                    )

                    return ast.copy_location(new_node, node)

                elif node.func.attr in ['assert_true', 'assert_false']:
                    attr = convert_to_camelcase(node.func.attr)
                    new_node = ast.Call(
                                func=ast.Attribute(
                                    value=ast.Attribute(
                                        value=ast.Name(id='nbtest', ctx=ast.Load()),
                                        attr='tc',
                                        ctx=ast.Load(),
                                    ),
                                    attr=attr,
                                    ctx=ast.Load()
                                ),
                                args=node.args,
                                keywords=keywords
                            )

                    return ast.copy_location(new_node, node)

        return self.generic_visit(node)

def replace_nbtest(tree, node):
    transformer = ReplaceNbtest(node)
    new_tree = transformer.visit(tree)
    return new_tree

class NbCellError(Exception):
    """ custom exception for error reporting. """
    def __init__(self, cell_num, msg, source, traceback=None, *args, **kwargs):
        self.cell_num = cell_num
        super(NbCellError, self).__init__(msg, *args, **kwargs)
        self.source = source
        self.inner_traceback = traceback

def pytest_addoption(parser):
    """
    Adds the --nbtest option flags for pytest.

    Adds an optional flag to pass a config file with regex
    expressions to sanitise the outputs
    Only will work if the --nbtest flag is present

    This is called by the pytest API
    """
    group = parser.getgroup("nbtest", "Jupyter Notebook testing")

    group.addoption('--nbtest', action='store_true',
                    help="Run Jupyter notebooks, validating all output")

    group.addoption('--nbtest-seed', action='store', default=-1,
                    type=int,
                    help='Seed to be used for input notebook')

    group.addoption('--nbtest-output-dir', action='store', default="./",
                    help='Path to store pytest logs')

    group.addoption('--nbtest-log-filename', action='store', default="test_log.csv",
                    help='Name of csv file to store pytest logs')

    group.addoption('--nbtest-nblog-name', action='store', default="test_nblog.ipynb",
                    help='Name of notebook to store tracebacks in')

@pytest.fixture
def change_test_dir(request):
    os.chdir(request.fspath.dirname)

def pytest_collect_file(file_path, parent):
    """
    Collect IPython notebooks using the specified pytest hook
    """
    opt = parent.config.option
    if (opt.nbtest) and file_path.suffix == ".ipynb":
        # os.chdir(request.fspath.dirname)
        return IPyNbFile.from_parent(parent, path=file_path)

class IPyNbFile(pytest.File):
    """
    This class represents a pytest collector object.
    A collector is associated with an ipynb file and collects the cells
    in the notebook for testing.
    yields pytest items that are required by pytest.
    """
    def __init__(self, *args, **kwargs):
        super(IPyNbFile, self).__init__(*args, **kwargs)
        config = self.parent.config
        self.sanitize_patterns = OrderedDict()  # Filled in setup_sanitize_patterns()
        self.timed_out = False
        self.skip_compare = (
            'metadata',
            'traceback',
            'prompt_number',
            'output_type',
            'name',
            'execution_count',
            'application/vnd.jupyter.widget-view+json'  # Model IDs are random
        )

    kernel = None

    def setup(self):
        """
        Called by pytest to setup the collector cells in
        Here we start a kernel
        """
        kernel_name = self.nb.metadata.get(
                'kernelspec', {}).get('name', 'python')
        self.kernel = RunningKernel(
            kernel_name,
            # cwd=str(self.fspath.dirname),
            cwd = str(os.getcwd())
        )

    def get_kernel_message(self, timeout=None, stream='iopub'):
        """
        Gets a message from the iopub channel of the notebook kernel.
        """
        return self.kernel.get_message(stream, timeout=timeout)

    # Read through the specified notebooks and load the data
    # (which is in json format)
    def collect(self):
        """
        The collect function is required by pytest and is used to yield pytest
        Item objects. We specify an Item for each function starting with `nbtest.assert_`.
        """
        self.nb = nbformat.read(str(self.fspath), as_version=4)
        code_cell_idx = 0
        last_exec_cell = 0

        seed = self.config.option.nbtest_seed

        # Iterate over the cells in the notebook
        for cell_index, cell in enumerate(self.nb.cells):
            if cell.cell_type == 'code':

                if (seed != -1 and cell_index == 0 and 'print(seed)' in cell.source):
                    lines = cell.source.split('\n')
                    lines[1] = f'seed = {seed}'
                    cell.source = '\n'.join(lines) + '\n'
                    continue

                code_cell_idx += 1
                # Parse the cell's source code using ast
                try:
                    tree = ast.parse(cell.source)
                except:
                    continue

                asserts = 0
                # Count assertions
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'nbtest':
                        asserts += 1

                assert_idx = 0
                # Find nbtest.assert_*()
                last_exec_line = 0
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'nbtest':
                        assert_idx += 1

                        test_name = None
                        for kw in node.keywords:
                            if kw.arg == "test_id":
                                test_name = kw.value.value

                        if not test_name:
                            test_name = (f'{cell_index}_{node.lineno}')

                        yield IPyNbFunction.from_parent(
                            self, name=test_name, cell_num=cell_index, cells=self.nb.cells,
                            last_exec_cell = last_exec_cell, test_node=node, test_line_num = node.lineno - 1,
                            last_exec_line = last_exec_line, exec_remaining = bool(assert_idx == asserts)
                        )
                        last_exec_cell = cell_index + 1
                        last_exec_line = node.lineno

class IPyNbFunction(pytest.Item):
    def __init__(self, name, parent, cell_num, cells, last_exec_cell, test_node, test_line_num, last_exec_line, exec_remaining):
        super().__init__(name, parent)
        self.cell_num = cell_num
        self.cells = cells
        self.last_exec_cell = last_exec_cell
        self.test_node = test_node
        self.test_line_num = test_line_num
        self.last_exec_line = last_exec_line
        self.exec_remaining = exec_remaining
        self.colors = bcolors if self.config.option.color != 'no' else nocolors

    def raise_cell_error(self, message, cell_source, exec = False, curr_cell_lines = [], *args, **kwargs):
        kernel = self.parent.kernel
        if (exec):
            valid_cell = ""
            for i in range(self.test_line_num + 1, len(curr_cell_lines)):
                valid_cell += curr_cell_lines[i] + "\n"

            msg_id = kernel.execute_cell_input(valid_cell, allow_stdin=False)

            while True:
                try:
                    msg = self.parent.get_kernel_message(timeout=10000)
                except TimeoutError:
                    raise RuntimeError("Kernel message timeout exceeded")

                msg_type = msg['msg_type']
                reply = msg['content']

                if msg_type == 'status' and reply['execution_state'] == 'idle':
                    break

                if msg_type == 'error':
                    try:
                        self.parent.kernel.await_idle(msg_id, 100)
                    except Empty:
                        self.stop()
                        raise RuntimeError('Timed out waiting for idle kernel!')

                    traceback = '\n'.join(reply['traceback'])
                    err_tracebacks.append((self.cell_num, reply['traceback']))
                    self.raise_cell_error("Runtime error", traceback)

        raise NbCellError(self.cell_num, message, cell_source, *args, **kwargs)

    def repr_failure(self, excinfo):
        """ called when self.runtest() raises an exception. """
        exc = excinfo.value
        cc = self.colors
        if isinstance(exc, NbCellError):
            msg_items = [
                cc.FAIL + "Assertion failed" + cc.ENDC]
            formatstring = (
                cc.OKBLUE + "Cell %d: %s\n\n" +
                "Input:\n" + cc.ENDC + "%s\n")
            msg_items.append(formatstring % (
                exc.cell_num,
                str(exc),
                exc.source
            ))
            if exc.inner_traceback:
                msg_items.append((
                    cc.OKBLUE + "Traceback:" + cc.ENDC + "\n%s\n") %
                    exc.inner_traceback)
            return "\n".join(msg_items)
        else:
            return "pytest plugin exception: %s" % str(exc)

    def reportinfo(self):
        description = "%s::Cell %d" % (self.fspath.relto(self.config.rootdir), self.cell_num)
        return self.fspath, 0, description

    def runtest(self):
        kernel = self.parent.kernel
        if not kernel.is_alive():
            raise RuntimeError("Kernel dead on test start")

        global err_tracebacks

        # Execute the previous cells
        for i in range(self.last_exec_cell, self.cell_num):
            if (self.cells[i].cell_type == 'code'):
                msg_id = kernel.execute_cell_input(self.cells[i].source, allow_stdin=False)

                while True:
                    try:
                        msg = self.parent.get_kernel_message(timeout=10000)
                    except TimeoutError:
                        raise RuntimeError("Kernel message timeout exceeded")

                    msg_type = msg['msg_type']
                    reply = msg['content']

                    if msg_type == 'status' and reply['execution_state'] == 'idle':
                        break

                    if msg_type == 'error':
                        with open(os.path.join(self.config.option.nbtest_output_dir, self.config.option.nbtest_log_filename), '+a') as f:
                            f.write(f"{self.name},-1\n")
                        try:
                            self.parent.kernel.await_idle(msg_id, 100)
                        except Empty:
                            self.stop()
                            raise RuntimeError('Timed out waiting for idle kernel!')
                        traceback = '\n'.join(reply['traceback'])
                        err_tracebacks.append((self.cell_num, reply['traceback']))
                        self.raise_cell_error("Runtime error", traceback)

        curr_cell_lines = self.cells[self.cell_num].source.split('\n')
        valid_cell = ""
        for i in range(self.last_exec_line, self.test_line_num):
            valid_cell += curr_cell_lines[i] + "\n"

        msg_id = kernel.execute_cell_input(valid_cell, allow_stdin=False)

        while True:
            try:
                msg = self.parent.get_kernel_message(timeout=10000)
            except TimeoutError:
                raise RuntimeError("Kernel message timeout exceeded")

            msg_type = msg['msg_type']
            reply = msg['content']

            if msg_type == 'status' and reply['execution_state'] == 'idle':
                break

            if msg_type == 'error':
                with open(os.path.join(self.config.option.nbtest_output_dir, self.config.option.nbtest_log_filename), '+a') as f:
                    f.write(f"{self.name},-1\n")
                try:
                    self.parent.kernel.await_idle(msg_id, 100)
                except Empty:
                    self.stop()
                    raise RuntimeError('Timed out waiting for idle kernel!')
                traceback = '\n'.join(reply['traceback'])
                err_tracebacks.append((self.cell_num, reply['traceback']))
                self.raise_cell_error("Runtime error", traceback)

        # Execute the assertion in the current cell
        tree = ast.parse(self.test_node)
        new_tree = replace_nbtest(tree, self.test_node)

        msg_id = kernel.execute_cell_input(ast.unparse(new_tree), allow_stdin=False)

        while True:
            try:
                msg = self.parent.get_kernel_message(timeout=10000)
            except TimeoutError:
                raise RuntimeError("Kernel message timeout exceeded")

            msg_type = msg['msg_type']
            reply = msg['content']

            if msg_type == 'status' and reply['execution_state'] == 'idle':
                break

            if msg_type == 'error':
                with open(os.path.join(self.config.option.nbtest_output_dir, self.config.option.nbtest_log_filename), '+a') as f:
                    f.write(f"{self.name},0\n")
                try:
                    self.parent.kernel.await_idle(msg_id, 100)
                except Empty:
                    self.stop()
                    raise RuntimeError('Timed out waiting for idle kernel!')

                traceback = '\n'.join(reply['traceback'])
                err_tracebacks.append((self.cell_num, reply['traceback']))

                self.raise_cell_error("Assertion error", traceback, self.exec_remaining, curr_cell_lines)

        if (self.exec_remaining):
            valid_cell = ""
            for i in range(self.test_line_num + 1, len(curr_cell_lines)):
                valid_cell += curr_cell_lines[i] + "\n"

            msg_id = kernel.execute_cell_input(valid_cell, allow_stdin=False)

            while True:
                try:
                    msg = self.parent.get_kernel_message(timeout=10000)
                except TimeoutError:
                    raise RuntimeError("Kernel message timeout exceeded")

                msg_type = msg['msg_type']
                reply = msg['content']

                if msg_type == 'status' and reply['execution_state'] == 'idle':
                    break

                if msg_type == 'error':
                    try:
                        self.parent.kernel.await_idle(msg_id, 100)
                    except Empty:
                        self.stop()
                        raise RuntimeError('Timed out waiting for idle kernel!')

                    traceback = '\n'.join(reply['traceback'])
                    err_tracebacks.append((self.cell_num, reply['traceback']))
                    self.raise_cell_error("Runtime error", traceback)

        with open(os.path.join(self.config.option.nbtest_output_dir, self.config.option.nbtest_log_filename), '+a') as f:
            f.write(f"{self.name},1\n")

@pytest.hookimpl()
def pytest_sessionfinish(session):
    try:
        # First try to get from command line args
        notebook_path = next(
            arg for arg in session.config.args
            if arg.endswith('.ipynb')
        )
    except StopIteration:
        return

    new_ntbk = nbformat.read(notebook_path, as_version=4)
    os.chdir(session.config.getoption("--nbtest-output-dir"))

    for e in err_tracebacks:
        err_output = new_output(
            output_type='error',
            ename='Error',
            evalue="Assertion error",
            traceback=e[1]
        )

        if not hasattr(new_ntbk.cells[e[0]], 'outputs'):
            new_ntbk.cells[e[0]].outputs = []

        new_ntbk.cells[e[0]].outputs.append(err_output)

    nb_log_name = str(session.config.getoption("--nbtest-nblog-name"))
    nbformat.write(new_ntbk, nb_log_name)
