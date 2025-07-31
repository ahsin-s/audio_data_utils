import os 
import asyncio
import argparse
from pathlib import Path 
from more_itertools import chunked
from typing import List, Dict, Any

import numpy as np 
import soundfile as sf
import librosa 
import tqdm

from helpers import find_files

class AudioDataUtility:
    def __init__(
        self, 
        source_directory: str, 
        output_directory: str, 
        file_extension: str,
        target_length: int, 
        target_samplerate: int,
        padding_strategy: str = "repeat",
        resume: bool = False,
        show_progress_bar: bool = True,
        concurrency_limit: int = 128,
        ):
        self.source_directory = source_directory
        self.output_directory = output_directory
        self.file_extension =  file_extension
        self.target_length = target_length 
        self.target_samplerate = target_samplerate
        self.resume = resume 
        self.show_progress_bar = show_progress_bar
        self.concurrency_limit = concurrency_limit

        if not padding_strategy in ["repeat", "zero_pad"]:
            raise ValueError("`padding_strategy` must be 'repeat' or 'zero_pad'")

        self.padding_strategy_func = {
            "repeat": self.repeat_audio,
            "zero_pad": self.zero_pad_audio
        }[padding_strategy]

        os.makedirs(output_directory, exist_ok=True)

        print("Searching for files in the source directory (this can take some time)")
        self.all_source_files = find_files(str(self.source_directory), file_extension, as_generator=False)
        # print(f"Found {len(self.all_source_files)} files")

        self.chunking_overlap_factor = 0.1
    
    def set_chunking_overlap_factor(self, factor: float): 
        self.chunking_overlap_factor = factor 


    def chunk_audio_to_target_length(self, audio_data, target_length) -> List[np.array]:
        audio_size = len(audio_data) 
        # chunk with some overlap 
        chunked_audio = []
        while len(audio_data) > target_length:
            chunk = audio_data[:target_length]
            chunked_audio.append(chunk)
            audio_data = audio_data[int(target_length*(1-self.chunking_overlap_factor))::]


        last_chunk = audio_data[-target_length::]
        chunked_audio.append(last_chunk)
        return chunked_audio
    
    def repeat_audio(self, audio_data):
        audio_size = len(audio_data)
        repeat_factor = int(np.ceil(self.target_length / audio_size))
        audio_data = np.tile(audio_data, repeat_factor)
        return audio_data[:self.target_length]

    def zero_pad_audio(self, audio_data):
        zeros_needed = self.target_length - len(audio_data)
        zeros_array = np.zeros(zeros_needed)
        audio_data = np.concat([audio_data, zeros_array])
        return audio_data

    def save_chunked_audio(self, filename, chunked_audio, samplerate) -> List[str]:
        output_filepaths = []
        if len(chunked_audio) == 1:
            # no need to modify the filename to include the chunk info 
            output_filepath = os.path.join(self.output_directory, filename) 
            sf.write(output_filepath, chunked_audio[0], samplerate)
            output_filepaths.append(output_filepath)
        else:
            for chunknum, audiochunk in enumerate(chunked_audio):
                # add the chunknum suffix to the original filename
                chunk_filename = Path(filename).stem + f"_part_{chunknum}" + Path(filename).suffix
                output_filepath = os.path.join(self.output_directory, chunk_filename) 
                sf.write(output_filepath, audiochunk, samplerate)
                output_filepaths.append(output_filepath)
        return output_filepaths

    def normalize_audio_size_maintain_samplerate(self):
        rechunked_filepaths_lookup = {}
        for filepath in self.maybe_show_progress(self.all_source_files):
            try:
                audio_data, samplerate = sf.read(filepath)
                if len(audio_data) > self.target_length:
                    # too short, need to apply padding or looping logic
                    audio_data = self.padding_strategy_func(audio_data)
                    chunked_audio = [audio_data]
                else:
                    # too long, need to chunk it while maintaining the same samplerate
                    chunked_audio = self.chunk_audio_to_target_length(audio_data, samplerate)
                
                # save 
                saved_to = self.save_chunked_audio(Path(filepath).name, chunked_audio, samplerate)
                rechunked_filepaths_lookup[filepath] = saved_to
            except Exception as e:
                print(f"Couldn't process {filepath}")
                print(e)

        print("Complete")

    async def resize_audio(self, input_filepath, output_filepath, semaphore):
        async with semaphore:
            try:
                x, sample_rate = sf.read(input_filepath)
                if x.ndim > 1:
                    x = np.mean(x, axis=1)  # Convert to mono
                    
                if sample_rate > self.target_samplerate:
                    # in this case, resample to downsize the audio
                    print(f"{input_filepath} samplerate: {sample_rate} ; length: {len(x)}, total_duration: {len(x) / sample_rate}")
                    x = await asyncio.to_thread(librosa.resample, x, orig_sr=sample_rate, target_sr=self.target_samplerate)
                current_samples=len(x)
                if current_samples < self.target_length: 
                    # too short - apply the padding logic
                    x = self.padding_strategy_func(x)
                x = x[:self.target_length]  # clips if the audio data is still too long
                sf.write(output_filepath, x, self.target_samplerate)
            except Exception as e:
                print(e)

    async def normalize_audio_resample_if_needed(self):
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        tasks = []
        for filepath in self.maybe_show_progress(self.all_source_files):
            if len(tasks) >= 1000:
                asyncio.gather(*tasks)
                tasks = []
            output_filepath = os.path.join(self.output_directory, Path(filepath).name)
            if os.path.exists(output_filepath) and not self.resume:
                # output path exists but resume is False so it will be overwritten
                tasks.append(self.resize_audio(filepath, output_filepath)) 
        # do any remaining tasks 
        asyncio.gather(*tasks) 
        print("Complete")
        
    def maybe_show_progress(self, iterable):
        if self.show_progress_bar:
            return tqdm.tqdm(iterable)
        return iterable


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--file_suffix", type=str, required=True)
    parser.add_argument("--normalization_type", type=str, choices=["resample", "resize"], 
        help="Choose 'resample' to modify the audio to a fixed sample rate with clipping, or 'resize' to maintain the sample rate while chunking the audio with overlap to a fixed size")
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--samplerate", type=int, required=True)
    parser.add_argument("--concurrency_limit", required=False, type=int, default=64)
    parser.add_argument("--resume", required=False, type=bool, default=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    target_length = args.duration * args.samplerate
    audio_data_utility = AudioDataUtility(
        args.source_directory,
        args.output_directory,
        args.file_suffix,
        target_length,
        args.samplerate,
        concurrency_limit=args.concurrency_limit,
        resume=args.resume,
    )
    if args.normalization_type == "resample":
        asyncio.run(audio_data_utility.normalize_audio_resample_if_needed())
    else:
        audio_data_utility.normalize_audio_size_maintain_samplerate()
