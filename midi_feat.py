import os
import pretty_midi
from tqdm import tqdm
from glob import glob

import input_output as io
from features import get_features
from configuration import config

composers = ['mozart', 'beethoven', 'scarlatti', 'haydn']


def main(midi_files):
    for midi_file in tqdm(midi_files):
        file_struct = io.FileStruct(midi_file=midi_file)

        pm = pretty_midi.PrettyMIDI(midi_file)
        y = pm.fluidsynth(fs=float(config.sample_rate))

        get_features(file_struct.audio_file, 'chroma',
                     config, 'framesync', y=y, save=True)

        get_features(file_struct.audio_file, 'mel',
                     config, 'framesync', y=y, save=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_path", help="Path to the dataset",
                        type=str, default='./data')

    args, _ = parser.parse_known_args()
    for composer in args.composers:
        midi_files = glob(os.path.join(args.ds_path,
                                       'midi',
                                       f"{composer}-*.mid"))
        print(f"Process {args.ds_path} {composer}")
        main(midi_files)
