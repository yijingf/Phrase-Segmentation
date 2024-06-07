import os
import torch
import pickle
import librosa
import numpy as np
from glob import glob

import input_output as io
from configuration import config
from models import EmbedNet
from embed import embed_tracks
from algorithms.scluster.main2 import do_segmentation as scluster
from utils import times_to_intervals, remove_empty_segments

composers = ['mozart', 'haydn', 'scarlatti', 'beethoven']


def post_process_boundaries(file_struct, est_inter_list, est_labels_list):
    beat_frames = io.read_beats(file_struct.beat_file)
    beat_frames = librosa.util.fix_frames(beat_frames)
    cleaned_predictions = []
    beat_times = librosa.frames_to_time(
        beat_frames, sr=config.sample_rate, hop_length=config.hop_length)
    for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
        est_idxs = [beat_times[int(i)] for i in est_idxs]
        est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
        est_inter = times_to_intervals(est_idxs)
        cleaned_predictions.append((est_inter, est_labels))
    return cleaned_predictions


def segment(config):

    # setting device
    use_cuda = torch.cuda.is_available() and not config.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load model
    embedding_net = EmbedNet(config).to(device)
    model_path = config.models_dir.joinpath(config.model_name + '.pt')
    embedding_net.load_state_dict(torch.load(model_path)['state_dict'])
    embedding_net.to(device)
    embedding_net.eval()
    print('Model loaded')

    # create embeddings folder if does not exist
    embeddings_folder = os.path.join(config.ds_path, 'features', 'embeddings')
    if not (os.path.exists(embeddings_folder)):
        os.mkdir(embeddings_folder)

    # create predictions folder if does not exist
    predictions_folder = os.path.join(config.ds_path, 'predictions')
    if not (os.path.exists(predictions_folder)):
        os.mkdir(predictions_folder)

    # build tracklist
    tracklist = []
    for composer in composers:
        filelist = sorted(glob(os.path.join(config.ds_path,
                                            f"features/{composer}*beat*")))
        for fname in filelist:
            fname = fname.replace('features', 'audio')
            fname = fname.replace('_beats_', '')
            fname = fname.replace('.json', '.wav')
            tracklist.append(fname)

    # calculate embeddings
    for st in range(0, len(tracklist), config.batch_size):
        ed = min(st + config.batch_size, len(tracklist))
        embeddings_list = embed_tracks(tracklist[st:ed],
                                       embedding_net, config, device)

        # calculate boundaries
        for (track, embeddings) in embeddings_list:
            file_struct = io.FileStruct(track)
            np.save(file_struct.embeddings_file, embeddings)
            est_inter_list, est_labels_list, _ = scluster(
                embeddings.T, embeddings.T, True)
            cleaned_predictions = post_process_boundaries(
                file_struct, est_inter_list, est_labels_list)
            pickle.dump(cleaned_predictions, open(
                file_struct.predictions_file, "wb"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--ds_path', type=str, help='Path to the dataset.')
    parser.add_argument('--model_name', type=str, help='Model name.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size, defaults to 16.')
    args, _ = parser.parse_known_args()

    config.model_name = args.model_name
    config.ds_path = args.ds_path
    config.batch_size = args.batch_size
    segment(config)
