import os 
import asyncio
import argparse
from pathlib import Path 
from more_itertools import chunked

import numpy as np 
import soundfile as sf
import librosa 
import tqdm

from helpers import find_files 


async def resize_audio(duration_seconds: int, expected_samplerate: int, input_filepath: str, output_filepath, semaphore):
    async with semaphore:
        try:
            target_length = duration_seconds*expected_samplerate
            x, sample_rate = sf.read(input_filepath)
            if x.ndim > 1:
                x = np.mean(x, axis=1)  # Convert to mono
                
            if sample_rate > expected_samplerate:
                # in this case, resample to downsize the audio
                print(f"{input_filepath} samplerate: {sample_rate} ; length: {len(x)}, total_duration: {len(x) / sample_rate}")
                x = await asyncio.to_thread(librosa.resample, x, orig_sr=sample_rate, target_sr=expected_samplerate)
            current_samples=len(x)
            if current_samples < target_length: 
                # apply the repeat logic
                repeat_factor = int(np.ceil(target_length / current_samples))
                x = np.tile(x, repeat_factor)
            x = x[:target_length]  # additional validation
            sf.write(output_filepath, x, expected_samplerate)
        except Exception as e:
            print(e)

async def run_resize_for_all_files_in_directory(source_directory, output_directory: str, file_suffix, concurrency_limit, duration: int, samplerate: int, resume: bool):
    output_directory = Path(output_directory)
    source_directory = Path(source_directory)
    os.makedirs(output_directory, exist_ok=True)

    all_files = find_files(str(source_directory), file_suffix)
    print(f"Found {len(all_files)} files")
    tasks = []
    semaphore = asyncio.Semaphore(concurrency_limit)
    for batch in tqdm.tqdm(chunked(all_files, 10000), total=len(all_files)//10000 + 1, desc="Processing"):
        for source_file in batch:
            relative_path = source_file.relative_to(source_directory)
            output_filepath = output_directory.joinpath(relative_path)
            if output_filepath.exists() and resume:
                # skip
                continue 
            tasks.append(resize_audio(
                duration,
                samplerate,
                source_file,
                output_filepath,
                semaphore
            )) 
        await asyncio.gather(*tasks)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--file_suffix", type=str, required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--samplerate", type=int, required=True)
    parser.add_argument("--concurrency_limit", required=False, type=int, default=64)
    parser.add_argument("--resume", required=False, type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    asyncio.run(run_resize_for_all_files_in_directory(
        args.source_directory,
        args.output_directory,
        args.file_suffix,
        args.concurrency_limit,
        args.duration,
        args.samplerate,
        args.resume
    ))