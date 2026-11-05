import os

SKIP = {"venv", ".venv", "__pycache__", ".git", "node_modules", ".idea", ".vscode"}

def show(path, prefix=""):
    items = sorted(i for i in os.listdir(path) if i not in SKIP)
    for n, name in enumerate(items):
        last = n == len(items) - 1
        print(prefix + ("└── " if last else "├── ") + name)
        full = os.path.join(path, name)
        if os.path.isdir(full):
            show(full, prefix + ("    " if last else "│   "))

print(os.path.basename(os.getcwd()) + "/")
show(".")