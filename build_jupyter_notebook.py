"""
Compiles and executes all 30 practical exercises from Module 1, Module 2, and Module 3
into a single unified, fully executed master Jupyter Notebook (.ipynb).
"""

import os
import ast
import nbformat
from nbclient import NotebookClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "Soft_Computing_Assignment_2_All_Practicals.ipynb")

MODULES = [
    ("Module 1: Fuzzy Logic and Systems", "Module_1_Fuzzy_Logic", 10),
    ("Module 2: Artificial Neural Networks (ANNs)", "Module_2_Neural_Networks", 10),
    ("Module 3: Genetic Algorithms & Evolutionary Computing", "Module_3_Genetic_Algorithms", 10)
]

def sanitize_code_for_notebook(code):
    code = code.replace("os.path.dirname(__file__)", "os.getcwd()")
    code = code.replace("matplotlib.use('Agg')", "# matplotlib.use('Agg')")
    code = code.replace("plt.close()", "plt.show()")
    code = code.replace("if __name__ == '__main__':", "if True:")
    code = code.replace('if __name__ == "__main__":', "if True:")
    return code

def extract_docstring_and_code(file_path):
    with open(file_path) as f:
        raw_content = f.read()

    tree = ast.parse(raw_content)
    docstring = ast.get_docstring(tree) or ""

    lines = raw_content.split("\n")
    if lines and lines[0].strip().startswith(('"""', "'''")):
        end_idx = 1
        while end_idx < len(lines):
            if lines[end_idx].strip().endswith(('"""', "'''")):
                end_idx += 1
                break
            end_idx += 1
        code = "\n".join(lines[end_idx:]).strip()
    else:
        code = raw_content

    return docstring, sanitize_code_for_notebook(code)

def build_and_execute_master_notebook():
    cells = []

    intro_md = (
        "# STDA2102: Soft Computing (EL1) - Practical Laboratory Record\n"
        "## Assignment 2: Practical Exercises (Complete 30 Practicals)\n\n"
        "**Student Name:** Manish Kumar  \n"
        "**Course Code:** STDA2102  \n"
        "**Evaluation Total:** 100 Marks (50 Marks CCE conversion)  \n"
        "**Modules Covered:**\n"
        "1. **Module 1:** Fuzzy Logic and Systems (Practicals 1.1 to 1.10)\n"
        "2. **Module 2:** Artificial Neural Networks (Practicals 2.1 to 2.10)\n"
        "3. **Module 3:** Genetic Algorithms & Evolutionary Computing (Practicals 3.1 to 3.10)\n\n"
        "---"
    )
    cells.append(nbformat.v4.new_markdown_cell(intro_md))

    setup_code = (
        "import os\n"
        "os.environ['KERAS_BACKEND'] = 'torch'\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n"
        "print('Global environment initialized successfully!')"
    )
    cells.append(nbformat.v4.new_code_cell(setup_code))

    for mod_title, mod_folder, count in MODULES:
        cells.append(nbformat.v4.new_markdown_cell(f"# {mod_title}\n---"))
        folder_path = os.path.join(BASE_DIR, mod_folder)

        for i in range(1, count + 1):
            prefix = f"ex{i:02d}"
            target_file = None
            for f in sorted(os.listdir(folder_path)):
                if f.startswith(prefix) and f.endswith(".py"):
                    target_file = f
                    break

            if not target_file:
                continue

            file_path = os.path.join(folder_path, target_file)
            docstring, code_text = extract_docstring_and_code(file_path)

            cells.append(nbformat.v4.new_markdown_cell(
                f"### Practical {mod_folder[-1]}.{i}: {docstring.splitlines()[0] if docstring else target_file}\n\n"
                f"```\n{docstring}\n```"
            ))
            cells.append(nbformat.v4.new_code_cell(code_text))

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.11"},
        "kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"}
    }

    print("Executing master all-in-one notebook (30 practicals)...")
    client = NotebookClient(nb, timeout=800, kernel_name="python3", resources={'metadata': {'path': BASE_DIR}})
    executed_nb = client.execute()

    with open(NOTEBOOK_PATH, "w") as out_f:
        nbformat.write(executed_nb, out_f)

    print(f"\n[✓] Master Jupyter Notebook executed & saved: {NOTEBOOK_PATH} ({os.path.getsize(NOTEBOOK_PATH)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    build_and_execute_master_notebook()
