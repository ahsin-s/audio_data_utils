from typing import List 
from pathlib import Path 


def find_files(root_directory, file_suffix) -> List[Path]:
    if not file_suffix.startswith("."):
        file_suffix = "." + file_suffix
    p = Path(root_directory)
    globbed = list(p.glob(f"**/*{file_suffix}"))

    return globbed
