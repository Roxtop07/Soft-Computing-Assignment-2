"""
Generates and executes standalone, production-grade Jupyter Notebooks (.ipynb)
for every module in the Soft Computing course:
- Assignment 1: Case Study (Intelligent Air Conditioning Fuzzy Controller)
- Assignment 2: Module 1 (Fuzzy Logic and Systems - 10 Practicals)
- Assignment 2: Module 2 (Artificial Neural Networks - 10 Practicals)
- Assignment 2: Module 3 (Genetic Algorithms & Evolutionary Computing - 10 Practicals)

Each notebook is executed using nbclient so all outputs and inline plots are pre-rendered!
"""

import os
import ast
import nbformat
from nbclient import NotebookClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
A2_DIR = BASE_DIR

def sanitize_code_for_notebook(code):
    # 1. Replace __file__ references
    code = code.replace("os.path.dirname(__file__)", "os.getcwd()")
    # 2. Enable inline plotting
    code = code.replace("matplotlib.use('Agg')", "# matplotlib.use('Agg')")
    code = code.replace("plt.close()", "plt.show()")
    # 3. Handle __name__ == '__main__' -> unindent the main block or replace
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

def create_executed_notebook(cells, output_paths, kernel_cwd):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.11"},
        "kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"}
    }

    target_name = os.path.basename(output_paths[0])
    print(f"\nExecuting notebook for target: {target_name} in {kernel_cwd}...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={'metadata': {'path': kernel_cwd}})
    executed_nb = client.execute()

    for out_p in output_paths:
        with open(out_p, "w") as f:
            nbformat.write(executed_nb, f)
        print(f"[✓] Saved executed notebook: {out_p} ({os.path.getsize(out_p)/1024:.1f} KB)")

# =====================================================================
# 1. Assignment 1: Case Study Notebook
# =====================================================================
def build_assignment_1_notebook():
    ac_script_path = os.path.join(A1_DIR, "ac_fuzzy_simulation.py")
    docstring, ac_code = extract_docstring_and_code(ac_script_path)

    cells = [
        nbformat.v4.new_markdown_cell(
            "# STDA2102: Soft Computing (EL1) - Assignment 1\n"
            "## Case Study: Design of an Intelligent Air Conditioning System Using Fuzzy Logic\n\n"
            "**Course Code:** STDA2102  \n"
            "**Student Name:** Manish Kumar  \n"
            "**Max Marks:** 50 Marks (10% Total CCE)  \n"
            "**Evaluation Criteria:** Research (15 Pts), Understanding (10 Pts), Explanation & Analysis (10 Pts), Presentation (15 Pts)\n\n"
            "### Abstract & Problem Statement\n"
            "Conventional on-off (Bang-Bang) thermostats suffer from high in-rush currents, significant temperature cycling (±2°C), "
            "and cannot natively compensate for humidity. This notebook implements an intelligent Mamdani Fuzzy Logic Controller (FLC) "
            "that coordinates indoor ambient temperature and relative humidity to modulate continuous compressor speed, achieving **24.23% energy savings**."
        ),
        nbformat.v4.new_code_cell(
            "import os\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "%matplotlib inline\n"
            "print('Environment initialized.')"
        ),
        nbformat.v4.new_markdown_cell(
            "### Full 24-Hour Thermodynamic Simulation & Control Surface Model\n"
            "Below is the complete simulation script defining fuzzy sets, the 15-rule Mamdani knowledge base, centroid defuzzification, "
            "and the 24-hour dynamic thermal ODE comparison against Bang-Bang and PID controllers."
        ),
        nbformat.v4.new_code_cell(ac_code)
    ]

    out_paths = [os.path.join(A1_DIR, "Assignment_1_Case_Study_Simulation.ipynb")]
    create_executed_notebook(cells, out_paths, A1_DIR)


# =====================================================================
# 2. Assignment 2: Module 1, Module 2, Module 3 Notebooks
# =====================================================================
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

    # Save in both module directory and practicals root directory for convenience
    nb_name = f"{mod_folder_name}.ipynb"
    out_paths = [
        os.path.join(A2_DIR, nb_name),
        os.path.join(mod_dir, nb_name)
    ]
    create_executed_notebook(cells, out_paths, mod_dir)

if __name__ == "__main__":
    print("Building and executing all individual module Jupyter Notebooks...")
    # Assignment 2 Modules
    modules = [
        ("Module 1: Fuzzy Logic and Systems", "Module_1_Fuzzy_Logic", 10),
        ("Module 2: Artificial Neural Networks (ANNs)", "Module_2_Neural_Networks", 10),
        ("Module 3: Genetic Algorithms & Evolutionary Computing", "Module_3_Genetic_Algorithms", 10)
    ]
    for title, folder, count in modules:
        build_module_notebook(title, folder, count)

    print("\nAll module notebooks successfully built and executed!")
