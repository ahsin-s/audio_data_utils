from typing import List 
from pathlib import Path 


def find_files(root_directory, file_suffix, as_generator: bool=False) -> List[Path]:
    if not file_suffix.startswith("."):
        file_suffix = "." + file_suffix
    p = Path(root_directory)
    globbed = p.glob(f"**/*{file_suffix}")
    if not as_generator:
        globbed = list(globbed)

    return globbed
