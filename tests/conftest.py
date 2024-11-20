import os
import sys

# Ensure app directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Register the persistence marker
def pytest_configure(config):
    config.addinivalue_line("markers", "persistence: mark test as a persistence test")
