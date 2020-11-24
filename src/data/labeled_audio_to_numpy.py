from tqdm import tqdm
import os
import librosa
import numpy as np
import argparse
from multiprocessing import Pool
from p_tqdm import p_map
import pickle
import argparse

BASE_PATH = "/media/aneesh/USB1000/UrbanSound8K/audio"
SEGMENT_DIR = "fold"
FILENAMES = [
    [
        os.path.join(BASE_PATH,f"{SEGMENT_DIR}{fold}", file_)
        for file_ in os.listdir(os.path.join(BASE_PATH,f"{SEGMENT_DIR}{fold}"))
        if ".wav" in file_
    ] for fold in range(1,11)
]

def save_features(
    filename,features_dir):
    y, sr = librosa.load(filename)
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T, axis=0)
    melspectrogram = np.mean(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36, fmax=8000).T, axis=0
    )
    chroma_stft = np.mean(
        librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=36).T, axis=0
    )
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=36).T, axis=0)
    chroma_cens = np.mean(
        librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=36).T, axis=0
    )
    melspectrogram.shape, chroma_stft.shape, chroma_cq.shape, chroma_cens.shape, mfccs.shape
    features = np.reshape(
        np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)),
        (36, 5, 1),
    )
    filename_without_path  = filename.split("/")[-1][:-4]
    # print(filename_without_path)
    np.save(os.path.join(features_dir,filename_without_path) ,features)
    # return features

def convert_fold(fold):
    features_dir = os.path.join(BASE_PATH, "features",f"fold{fold}")
    for f in tqdm(FILENAMES[fold]):
        save_features(f,features_dir)
    print(f"fold {fold} completed!")

def main():
    parser = argparse.ArgumentParser(
        description="Calculates the features and saves them as numpy file."
    )
    parser.add_argument("-cpus", type=int, help="number of cpus to use.")
    args = parser.parse_args()

    

    for i in range(int(10/args.cpus)+1):
        # one fold per cpu core.
        fold_index = range(max(i*args.cpus, 1),min((i+1)*args.cpus, 10))
        print(f"Preparing {fold_index}")
        with Pool(processes=len(fold_index) ) as pool:
            pool.map(convert_fold,fold_index)
        # non_silent_files.append(temp)

main()