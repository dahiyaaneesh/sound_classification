from tqdm import tqdm
import os
import librosa
import numpy as np
import argparse
from multiprocessing import Pool
from p_tqdm import p_map
import pickle
import argparse
from scipy.io import wavfile
import pandas as pd

BASE_PATH = "/media/aneesh/USB1000/Zurich_Urban_Sounds"
RECORDER = "TASCAM_RECORDER"
SEGMENT_DIR = "audio_segments"
FILENAMES = sorted(
        [
            f
            for f in os.listdir(os.path.join(BASE_PATH, RECORDER, SEGMENT_DIR))
            if ".wav" in f
        ]
    )
MAX_AUDIO = 13 # number of homogenous long audio recordings.
MIN_AUDIO = 2

def get_non_silent_segments(audio_index, factor=1.5):
    filenames =  [f for f in FILENAMES if (f"{audio_index}_"==f[:len(f"{audio_index}_")])]
    mean_segment= []
    for fil in tqdm(filenames):
        f, x = wavfile.read(os.path.join(BASE_PATH, RECORDER,SEGMENT_DIR,fil))
        mean_segment.append(np.mean(x))
    mean_segment = np.array(mean_segment)
    mean = np.mean(mean_segment)
    std_dev = np.std(mean_segment)
    non_silent_file_index = np.argwhere(np.abs(mean_segment - mean)>(factor*std_dev))
    return (np.array(filenames)[non_silent_file_index]).reshape((-1,))


def main():
    parser = argparse.PARSER
    parser = argparse.ArgumentParser(
        description="Prepares dataset with name of non silent segements based on abnomral means."
    )
    parser.add_argument("-cpus", type=int, help="number of cpus to use.", default=2)
    parser.add_argument("-std_factor", type=float, help="STD above which sound is considered non silent",default=1.5)
    args = parser.parse_args()
    filenames = sorted(
        [
            f
            for f in os.listdir(os.path.join(BASE_PATH, RECORDER, SEGMENT_DIR))
            if ".wav" in f
        ]
    )
    # print(get_non_silent_segments(3,1.5))
    non_silent_files = []
    for i in range(int(MAX_AUDIO/args.cpus)+1):
        # one file per cpu core.
        audio_index = range(max(i*args.cpus, MIN_AUDIO),min((i+1)*args.cpus, MAX_AUDIO))
        print(f"Preparing {audio_index}")
        with Pool(processes=len(audio_index) ) as pool:
            temp = pool.map(get_non_silent_segments,audio_index)
        non_silent_files.append(temp)
    
    data = pd.DataFrame({"Non_silent_segments":[item for array_list in non_silent_files for array in array_list for item in array]})
    data.to_csv(os.path.join(BASE_PATH, RECORDER, "non_silent_segment.csv"))
    # print([item for array_list in non_silent_files for array in array_list for item in array])


main()