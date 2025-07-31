import os 
import subprocess 
import tqdm 
import soundfile 
import shutil
import asyncio 
import traceback 
import argparse 

from helpers import find_files

def fix_corrupt_flac(input_filepath: str, output_filepath: str, overwrite=True):
    if os.path.exists(output_filepath):
        if overwrite:
            os.remove(output_filepath)
        else:
            raise Exception("Output destination exists and overwite_existing is False")
    # Validate output path
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try: 
        data, _ = soundfile.read(input_filepath)  # file is good
        shutil.copy(input_filepath, output_filepath)
        return 
    except Exception as e:
        command = ["ffmpeg", "-i", input_filepath, "-c:a", "flac", "-compression_level", "5", "-y", output_filepath]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except Exception as e:
            print(f"Couldn't process {input_filepath}")
            traceback.format_exc() 

def fix_corrupt_wav(input_filepath, output_filepath, overwrite=True):
    raise NotImplementedError

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--file_suffix", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    tasks = []
    for file in tqdm.tqdm(find_files(args.source_directory, args.file_suffix)):
        file = str(file)
        output_path = os.path.join(args.output_directory, file.name)
        if file_suffix.endswith("flac"):
            fix_corrupt_flac(file, output_path)

        elif file_suffix.endswith("wav"):
            fix_corrupt_wav(file, output_path)
        else:
            raise NotImplementedError(f"Handler to fix {args.file_suffix} not implemented")