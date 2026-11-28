"""Load the project's kit.env into os.environ when a kit Python tool runs directly (python3 kit/tools/x.py),
as lib.sh does for the bash tools. Values already in the environment win. KIT_NO_ENV=1 turns it off."""
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_kit_env(project_root=PROJECT_ROOT):
    path = os.path.join(project_root, "kit.env")
    if os.environ.get("KIT_NO_ENV") or not os.path.isfile(path):
        return
    # bash reads it, so $HOME, quotes and comments mean what they mean in lib.sh.
    dumped = subprocess.run(["bash", "-c", 'set -a; source "$1"; env -0', "kit_env", path],
                            cwd=project_root, capture_output=True, check=True).stdout
    for item in dumped.split(b"\0"):
        name, _, value = item.partition(b"=")
        if name.startswith(b"KIT_"):
            os.environ.setdefault(name.decode(), value.decode())
