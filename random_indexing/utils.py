# Olha Svezhentseva
# 10.03.2023

import os
import pickle
from typing import Any, Optional


def save_data(data: Any, filename: str, base_dir: Optional[str] = "random_indexing") -> None:
    """The function saves data into a file."""
    if base_dir is not None:
        filename = os.path.join(base_dir, filename)
    with open(filename, mode="wb") as file:
        pickle.dump(data, file)
        print(f'Data saved in {filename}')
        return

def load_data(filename: str, base_dir: Optional[str] = "random_indexing") -> Any:
    """The function loads pre-saved data."""
    if base_dir is not None:
        filename = os.path.join(base_dir, filename)
    with open(filename, "rb") as file:
        output = pickle.load(file)
    print(f'Loading from {filename}...')
    return output