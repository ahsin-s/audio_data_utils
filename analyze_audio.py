import os 
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import tqdm

import soundfile as sf


def get_audiofile_size_and_samplerate(path_to_audio_file: str):
    try:
        data, samplerate = sf.read(path_to_audio_file)
        return len(data), samplerate 
    except Exception as e:
        print(f"Couldn't open {path_to_audio_file}")
        print(e)
    return -1, -1

def get_stats(data: List[int]):
    mean = np.mean(data)
    median = np.median(data)
    largest = max(data) if data else 0
    smallest = min(data) if data else 0
    return f"Mean: {mean}, Median: {median}, Max: {largest}, Smallest: {smallest}"

def plot_distribution(data: List[int], title: str,  figsave_path:str,):
    plt.hist(data, bins=10)
    plt.title(title)

    plt.savefig(figsave_path)


def run_for_all_files(data_source_dir: str, labels_path: str, label_column: str, filename_column: str):
    labels_df = pd.read_csv(labels_path, sep=" ")
    print(f"Processing {labels_df.shape[0]} files")


    metadata = {}
    samplerate_metadata = []
    audio_size_metadata = []
    duration_metadata = []
    label_level_size_metadata = {"fake": [], "real": []}
    label_level_samplerate_metadata = {"fake": [], "real": []}

    for filename, label in tqdm.tqdm(zip(labels_df[filename_column], labels_df[label_column])):
        filepath_abs = os.path.join(data_source_dir, filename) 
        audio_size, audio_samplerate = get_audiofile_size_and_samplerate(filepath_abs)
        
        if audio_size >0 and  audio_samplerate >0:
            duration = audio_size / audio_samplerate
            metadata[filename] = {"size": audio_size, "samplerate": audio_samplerate}
            samplerate_metadata.append(audio_samplerate)
            audio_size_metadata.append(audio_size)
            duration_metadata.append(duration)

            label_level_samplerate_metadata[label].append(audio_samplerate)
            label_level_size_metadata[label].append(audio_size)

    
    print(f"\n\nAudio Size Stats:{get_stats(audio_size_metadata)}\n\n")
    print(f"\n\nSamplerate Stats:{get_stats(samplerate_metadata)}\n\n")
    print(f"\n\nDuration Stats:{get_stats(duration_metadata)}\n\n")

    title = Path(labels_path).stem
    plot_distribution(samplerate_metadata, title, f"samplerate_distribution_for_{title}.png")
    plot_distribution(audio_size_metadata, title, f"audiosize_distribution_for_{title}.png")


    print(f"\n\nAudio size for label 'fake':{get_stats(label_level_size_metadata['fake'])}")
    print(f"\n\nAudio size for 'real':{get_stats(label_level_size_metadata['real'])}")

    print(f"\n\nAudio samplerate for 'fake':{get_stats(label_level_samplerate_metadata['fake'])}")
    print(f"\n\nAudio samplerate for 'real':{get_stats(label_level_samplerate_metadata['real'])}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_directory", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--label_column", type=str, required=True)
    parser.add_argument("--filename_column", type=str, required=True)
    parser.add_argument("--concurrency_limit", required=False, type=int, default=64)
    parser.add_argument("--resume", required=False, type=bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args() 

    run_for_all_files(
        args.source_directory,
        args.labels_path,
        args.label_column,
        args.filename_column
    )

