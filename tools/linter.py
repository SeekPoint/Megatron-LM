import os
import os.path as osp
import pathlib
import subprocess
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")
gd.debuginfo(prj="mt")

def recursively_lint_files():
    """Recursively lint all python files in chosen subdirectories of megatron-lm"""

    try:
        import autopep8
    except ModuleNotFoundError:
        gd.debuginfo(prj="mt", info=f"Please first install autopep8 via `pip install autopep8`")
        return

    # get all python file paths from top level directory
    file_dir = str(pathlib.Path(__file__).parent.absolute())
    working_dir = osp.join(file_dir, os.pardir)
    all_py_paths = set(os.path.join(working_dir, fname)
                       for fname in os.listdir(working_dir) if ".py" in fname)

    # get all python file paths from chosen subdirectories
    check_dirs = ['docker', 'megatron', 'openwebtext', 'scripts', 'tasks']
    for sub_dir in check_dirs:
        for path, _, fnames in os.walk(osp.join(working_dir, sub_dir)):
            all_py_paths.update(set(osp.join(path, fname) for fname in fnames if ".py" in fname))

    gd.debuginfo(prj="mt", info=f"Linting the following: ")
    for py_path in all_py_paths:
        gd.debuginfo(prj="mt", info=fpy_path)
        command = 'autopep8 --max-line-length 100 --aggressive --in-place {}'.format(py_path)
        subprocess.check_call(command)


if __name__ == "__main__":
    recursively_lint_files()
