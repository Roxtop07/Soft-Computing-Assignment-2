"""
Generates and executes standalone, production-grade Jupyter Notebooks (.ipynb)
for Assignment 2 in the Soft Computing course:
- Module 1 (Fuzzy Logic and Systems - 10 Practicals)
- Module 2 (Artificial Neural Networks - 10 Practicals)

Each notebook is executed using nbclient so all outputs and inline plots are pre-rendered!
"""

import os
import ast
import nbformat
from nbclient import NotebookClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
A2_DIR = BASE_DIR

def sanitize_code_for_notebook(code):
    code = code.replace("os.path.dirname(__file__)", "os.getcwd()")
    code = code.replace("matplotlib.use('Agg')", "# matplotlib.use('Agg')")
    code = code.replace("plt.close()", "plt.show()")
    code = code.replace("if __name__ == '__main__':", "if True:")
    code = code.replace("if __name__ == \"__main__\":", "if True:")
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

def create_executed_notebook(cells, output_paths, kernel_cwd):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.11"},
        "kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"}
    }

    target_name = os.path.basename(output_paths[0])
    print(f"\nExecuting notebook for target: {target_name} in {kernel_cwd}...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": kernel_cwd}})
    executed_nb = client.execute()

    for out_p in output_paths:
        with open(out_p, "w") as f:
            nbformat.write(executed_nb, f)
        print(f"[✓] Saved executed notebook: {out_p} ({os.path.getsize(out_p)/1024:.1f} KB)")

def build_module_notebook(mod_title, mod_folder_name, count):
    mod_dir = os.path.join(A2_DIR, mod_folder_name)

    intro_md = (
        f"# STDA2102: Soft Computing (EL1) - Assignment 2\n"
        f"## {mod_title} (Complete 10 Practical Exercises)\n\n"
        f"**Course Code:** STDA2102  \n"
        f"**Student Name:** Manish Kumar  \n"
        f"**Practical Evaluation:** Implementation (15), Understanding (10), Code Quality (10), Analysis (10), Record (5) -> Total 50 Pts\n\n"
        f"---"
    )

    setup_code = (
        "import os\n"
        "os.environ['KERAS_BACKEND'] = 'torch'\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n"
        "print('Module environment configured successfully.')"
    )

    cells = [
        nbformat.v4.new_markdown_cell(intro_md),
        nbformat.v4.new_code_cell(setup_code)
    ]

    for i in range(1, count + 1):
        prefix = f"ex{i:02d}"
        target_file = None
        for f in sorted(os.listdir(mod_dir)):
            if f.startswith(prefix) and f.endswith(".py"):
                target_file = f
                break

        if not target_file:
            continue

        file_path = os.path.join(mod_dir, target_file)
        docstring, code_text = extract_docstring_and_code(file_path)

        cells.append(nbformat.v4.new_markdown_cell(
            f"## Practical {i}: {docstring.splitlines()[0] if docstring else target_file}\n\n"
            f"```\n{docstring}\n```"
        ))
        cells.append(nbformat.v4.new_code_cell(code_text))

    nb_name = f"{mod_folder_name}.ipynb"
    out_paths = [
        os.path.join(A2_DIR, nb_name),
        os.path.join(mod_dir, nb_name)
    ]
    create_executed_notebook(cells, out_paths, mod_dir)

if __name__ == "__main__":
    print("Building and executing all individual module Jupyter Notebooks...")
    modules = [
        ("Module 1: Fuzzy Logic and Systems", "Module_1_Fuzzy_Logic", 10),
        ("Module 2: Artificial Neural Networks (ANNs)", "Module_2_Neural_Networks", 10)
    ]
    for title, folder, count in modules:
        build_module_notebook(title, folder, count)

    print("\nAll module notebooks successfully built and executed!")
