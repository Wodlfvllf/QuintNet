import os
import pkgutil
import importlib
import pytest
from pathlib import Path

def get_all_python_files(root_dir):
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and typical build/cache dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('build', 'dist', '__pycache__', 'egg-info')]
        
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files

def path_to_module(path, root_dir):
    rel_path = os.path.relpath(path, root_dir)
    if rel_path.startswith('..'):
        return None # Outside of root
    
    # Remove extension
    module_path = os.path.splitext(rel_path)[0]
    
    # Replace separators with dots
    module_name = module_path.replace(os.path.sep, '.')
    
    # Handle __init__
    if module_name.endswith('.__init__'):
        module_name = module_name[:-9]
    
    return module_name

# Determine the project root. 
# Assuming this test file is in tests/ and the project root is one level up.
PROJECT_ROOT = Path(__file__).parent.parent

ALL_FILES = get_all_python_files(PROJECT_ROOT)

@pytest.mark.parametrize("file_path", ALL_FILES)
def test_import_file(file_path):
    """
    Attempts to import the given file as a module.
    """
    module_name = path_to_module(file_path, PROJECT_ROOT)
    
    if not module_name:
        pytest.skip(f"Could not determine module name for {file_path}")
        
    print(f"Attempting to import: {module_name} from {file_path}")
    
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {e}")
    except Exception as e:
        pytest.fail(f"An error occurred while importing {module_name}: {e}")
