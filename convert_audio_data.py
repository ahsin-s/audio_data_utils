import os 
import logging
import asyncio
import argparse
from pathlib import Path  
from typing import List, Any

import soundfile, ffmpeg


def convert_to_other_format(input_file, output_file,  **ffmpeg_kwargs):
    try:
        stream = ffmpeg.input(input_file)
        stream = ffmpeg.output(stream, output_file, **ffmpeg_kwargs)
        ffmpeg.run(stream, quiet=True, capture_stdout=True, capture_stderr=True, overwrite_output=True)
        return True 
    except ffmpeg.Error as e:
        logging.error(f"Error {e}")
    return False 

async def convert_to_other_format_async(input_file, output_file, semaphore, **ffmpeg_kwargs):
    async with semaphore:
        convert_to_other_format(input_file, output_file, **ffmpeg_kwargs)


def find_files(root_directory, file_suffix) -> List[Path]:
    if not file_suffix.startswith("."):
        file_suffix = "." + file_suffix
    p = Path(root_directory)
    globbed = list(p.glob(f"**/*{file_suffix}"))

    return globbed


async def perform_conversion_concurrently(root_directory: str, output_directory: str, file_suffix: str, output_format: str, concurrency_limit, **ffmpeg_kwargs):
    if not output_format.startswith("."):
        output_format = "." + output_format

    # find the files 
    all_files = find_files(root_directory, file_suffix)
    print(f"Found {len(all_files)} files")

    os.makedirs(output_directory, exist_ok=True)

    # Setup tasks 
    tasks = []
    semaphore = asyncio.Semaphore(concurrency_limit)
    for file in all_files:
        tasks.append(
            convert_to_other_format_async(
                str(file),
                os.path.join(output_directory, file.with_suffix(output_format)),
                semaphore,
                **ffmpeg_kwargs
            )
        )
    print("Running conversion tasks asynchronously")
    
    await asyncio.gather(*tasks)


def get_args():
    def parse_dict(arg):
        """Convert a string representation of a dictionary into a Python dictionary."""
        try:
            # Use ast.literal_eval to safely evaluate the string as a Python expression
            result = ast.literal_eval(arg)
            if not isinstance(result, dict):
                raise ValueError("Argument must be a dictionary")
            return result
        except (ValueError, SyntaxError) as e:
            raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")
            
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_directory",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        required=True 
    )
    parser.add_argument(
        "--output_format",
        type=str,
        required=True
    )
    parser.add_argument(
        "--concurrency_limit",
        type=int,
        required=False
    )
    parser.add_argument(
        "--ffmpeg_kwargs",
        type=parse_dict,
        help="Dictionary in Python syntax (e.g., \"{'key1': 'value1', 'key2': 42}\")",
        required=False
    )
    return parser 


if __name__ == "__main__":
    args = get_args() 

    parsed = args.parse_args()

    source_directory = parsed.source_directory
    output_directory = parsed.output_directory
    file_suffix = parsed.file_suffix
    output_format = parsed.output_format
    concurrency_limit = parsed.concurrency_limit or 64
    ffmpeg_kwargs = parsed.ffmpeg_kwargs or {}

    asyncio.run(
        perform_conversion_concurrently(
        root_directory=source_directory,
        output_directory=output_directory,
        file_suffix=file_suffix,
        output_format=output_format,
        concurrency_limit=concurrency_limit,
        **ffmpeg_kwargs
        )
    )

