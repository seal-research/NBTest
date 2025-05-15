import logging
import re
import nbformat as nbf
import tokenize
from io import StringIO
from pathlib import Path


def setup_logger(log_file_path, logger_name, level=logging.DEBUG):
        
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger

def set_seed_random_state(code):
    # Step 4: Replace 'random_state' argument values with 'random_seed'

    random_state_pattern = r"(random_state\s*=\s*)\d+(?=\s*[),])"
    cleaned_code = re.sub(random_state_pattern, r"\1random_seed", code)

    return cleaned_code


def preprocess_code(code):
    """
    Preprocess the code by:
    1. Removing lines starting with '!', '%', or containing 'pip install'.
    2. Removing redundant os.walk blocks that print file paths.
    3. Removing all comments and blank lines.
    """
    # Step 1: Remove lines with '!', '%', or 'pip install'
    cleaned_lines = [
        line for line in code.splitlines()
        if not (line.strip().startswith('!') or line.strip().startswith('%') or 'pip install' in line)
    ]
    cleaned_code = "\n".join(cleaned_lines)

    # Step 2: Remove os.walk blocks
    os_walk_pattern = (
        r"for\s+[\w\d_]+,\s*_[^:]*in\s+os\.walk\([^)]+\):\s*\n"
        r"(?:\s*#.*\n)*"  # Optional comments
        r"\s*for\s+[\w\d_]+[^:]*:\s*\n"
        r"(?:\s*#.*\n)*"  # Optional comments
        r"\s*print\(.*\)\s*"
    )
    cleaned_code = re.sub(os_walk_pattern, '', cleaned_code, flags=re.MULTILINE)

    # Step 3: Remove comments and blank lines using tokenize
    final_lines = []
    for line in cleaned_code.splitlines():
        stripped_line = line.strip()
        # Exclude blank lines and comments, preserve other lines with original spaces
        if stripped_line and not stripped_line.startswith("#"):
            final_lines.append(line)
    cleaned_code = "\n".join(final_lines)
    

    # # Step 4: Replace 'random_state' argument values with 'random_seed'

    # random_state_pattern = r"(random_state\s*=\s*)\d+(?=\s*[),])"
    # cleaned_code = re.sub(random_state_pattern, r"\1random_seed", cleaned_code)


    return cleaned_code

def get_notebook_name(notebook_path):
    notebook_stat_name = Path(notebook_path).stem
    notebook_fname = notebook_stat_name.split("_chebyshev")[0]
    return notebook_fname

def main():

    ntbk = nbf.read("/home/yy2282/project/nb_test/results/results/MAIN__chebyshev_2_0.95/kaggle__brokerus__ml-project-house-prices-eda-and-7-models/ml-project-house-prices-eda-and-7-models.ipynb", nbf.NO_CONVERT)
    

    for i, cell in enumerate(ntbk.cells):
        if cell.cell_type == "code":
            cleaned_code = preprocess_code(cell.source)
            print(cleaned_code)

    
   
if __name__ == "__main__":
    main()