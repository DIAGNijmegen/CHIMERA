import os

def generate_directory_tree(startpath):
    """
    Generates and prints a visual directory tree structure.

    Args:
        startpath (str): The absolute or relative path to the directory
                         you want to inspect.
    """
    # Check if the provided path is a valid directory
    if not os.path.isdir(startpath):
        print(f"Error: The path '{startpath}' is not a valid directory.")
        return
    
    print(f"📁 {os.path.basename(os.path.abspath(startpath))}")

    # The recursive helper function to build the tree
    _tree_helper(startpath)


def _tree_helper(directory, prefix=''):
    """
    A recursive helper function that does the heavy lifting of building the tree.

    Args:
        directory (str): The current directory being processed.
        prefix (str): The string of characters (like '│   ' and '    ')
                      that creates the visual tree structure.
    """
    try:
        # Get a list of all files and directories, ignoring hidden ones (e.g., .DS_Store)
        # os.listdir gives us everything in the current directory.
        nodes = [node for node in os.listdir(directory) if not node.startswith('.')]
        # Separate files and directories to process directories first
        files = sorted([n for n in nodes if os.path.isfile(os.path.join(directory, n))])
        dirs = sorted([n for n in nodes if os.path.isdir(os.path.join(directory, n))])
        # Combine them for printing, directories first
        entries = dirs + files
    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")
        return
    except FileNotFoundError:
        print(f"{prefix}└── [Directory Not Found]")
        return

    # These characters create the tree branches.
    # '├──' is for an intermediate entry.
    # '└──' is for the last entry in a directory.
    pointers = ['├── '] * (len(entries) - 1) + ['└── ']

    for i, entry in enumerate(entries):
        # Print the entry with its corresponding pointer and prefix
        print(f"{prefix}{pointers[i]}{entry}")

        # If the entry is a directory, we recurse into it
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            # The extension determines the prefix for the next level down.
            # If the current entry is the last one, we use empty space,
            # otherwise, we use a vertical bar to show the tree line continues.
            extension = '    ' if pointers[i] == '└── ' else '│   '
            _tree_helper(path, prefix=prefix + extension)


# --- CONFIGURATION ---
# ---------------------
# IMPORTANT: Change the path below to the directory you want to analyze.
#
# Examples:
# - On macOS/Linux: "/Users/your_username/Documents"
# - On Windows: "C:\\Users\\your_username\\Documents"
# - To use the current directory where the script is located, just use '.'
#
# ---------------------
PATH_TO_INSPECT = '/Users/robertspaans/Documents/Projects/phd_projects/multimodal_working_group/MICCAI2025/MICCAI2025_models/CHIMERA/task1_baseline'


# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    generate_directory_tree(PATH_TO_INSPECT)

