"""Build the C++ extension on first launch or after native source changes."""
from pathlib import Path
import subprocess
import sys
import sysconfig


def main():
    args = sys.argv[1:]
    engine = "cpp"
    for index, argument in enumerate(args):
        if argument == "--engine" and index + 1 < len(args):
            engine = args[index + 1]
        elif argument.startswith("--engine="):
            engine = argument.split("=", 1)[1]
    if engine != "cpp" or "--help" in args or "-h" in args:
        return
    root = Path(__file__).resolve().parent
    library = root / ("_motion_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    inputs = [root / "CMakeLists.txt", root / "build_native.sh", root / "requirements-build.txt"]
    inputs.extend(p for p in (root / "native").rglob("*") if p.is_file())
    if not library.is_file() or any(p.stat().st_mtime_ns > library.stat().st_mtime_ns for p in inputs):
        print("Building meganan C++ engine…", flush=True)
        subprocess.run([str(root / "build_native.sh")], cwd=root, check=True)


if __name__ == "__main__":
    main()
