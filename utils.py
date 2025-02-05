# -*- coding: utf-8 -*-
import os
import collections
import datetime

import ujson
import numpy as np
import torch
import librosa
import jams
import input_output as io


def clean_tracklist(config, annotations, tracklist=None):
    if tracklist is None:
        tracklist = librosa.util.find_files(os.path.join(
            config.ds_path, 'audio'), ext=config.dataset.audio_exts)
    tracklist_clean = []
    for song in tracklist:
        if valid_feat_files(io.FileStruct(song), 'mel', config) and os.path.isfile(io.FileStruct(song).beat_file) and os.path.isfile(io.FileStruct(song).json_file):
            if annotations and os.path.isfile(io.FileStruct(song).ref_file):
                tracklist_clean.append(song)
            else:
                tracklist_clean.append(song)
    return tracklist_clean


def get_ref_labels(file_struct, level, config, annot=0):
    if config.ds_path == None:
        config.ds_path = str(file_struct.ds_path)
    track = str(file_struct.audio_file)
    if 'SALAMI_2annot' in track or 'SALAMI_left' in track or 'SALAMI_test_MIREX' in track or 'SALAMI_pop' in track or 'SALAMI' in track:
        ref_inters_list, ref_labels_list, duration = io.read_references_2annot(
            file_struct.audio_file, annot)
        ref_times = intervals_to_times(ref_inters_list[level])
        ref_labels = ref_labels_list[level]
    elif 'JSD' in track:
        ref_times, ref_labels, duration = io.read_references_jsd(track, level)
    else:
        jam = jams.load(str(file_struct.ref_file), validate=False)
        duration = jam.file_metadata.duration
        ref_times, ref_labels = io.read_references(
            file_struct.audio_file, False)

    ref_times, ref_labels = remove_empty_segments(ref_times, ref_labels)
    ref_times[0] = 0
    return ref_labels, ref_times, duration


def rm_extension(fname):
    return os.path.splitext(fname)[0]


def save_model(directory, exp_config, epoch, model, optimizer):
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_config': str(exp_config)
                },
               directory.joinpath(exp_config.model_name + '.pt'))


def load_model(fname, device, model, optimizer, scheduler, *args):
    checkpoint = torch.load(fname,
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    infos = {}
    for arg in args:
        infos[arg] = checkpoint[arg]
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['optimizer_state_dict'])
    return infos


def times_to_intervals(times):
    """ Copied from MSAF.
    Given a set of times, convert them into intervals.
    Parameters
    ----------
    times: np.array(N)
        A set of times.
    Returns
    -------
    inters: np.array(N-1, 2)
        A set of intervals.
    """
    return np.asarray(list(zip(times[:-1], times[1:])))


def intervals_to_times(inters):
    """ Copied from MSAF.
    Given a set of intervals, convert them into times.
    Parameters
    ----------
    inters: np.array(N-1, 2)
        A set of intervals.
    Returns
    -------
    times: np.array(N)
        A set of times.
    """
    return np.concatenate((inters.flatten()[::2], [inters[-1, -1]]), axis=0)


def lognormalize(F, floor=0.1, min_db=-80):
    """ Copied from MSAF.
    Log-normalizes features such that each vector is between min_db to 0."""
    assert min_db < 0
    F = min_max_normalize(F, floor=floor)
    F = np.abs(min_db) * np.log10(F)  # Normalize from min_db to 0
    return F


def min_max_normalize(F, floor=0.001):
    """Copied from MSAF.
    Normalizes features such that each vector is between floor to 1."""
    F -= -F.min() + floor
    F /= F.max(axis=0)
    return F


def normalize(X, norm_type, floor=0.0, min_db=-80):
    """ Copied from MSAF.
    Normalizes the given matrix of features.
    Parameters
    ----------
    X: np.array
        Each row represents a feature vector.
    norm_type: {"min_max", "log", np.inf, -np.inf, 0, float > 0, None}
        - `"min_max"`: Min/max scaling is performed
        - `"log"`: Logarithmic scaling is performed
        - `np.inf`: Maximum absolute value
        - `-np.inf`: Mininum absolute value
        - `0`: Number of non-zeros
        - float: Corresponding l_p norm.
        - None : No normalization is performed
    Returns
    -------
    norm_X: np.array
        Normalized `X` according the the input parameters.
    """
    if isinstance(norm_type, str):
        if norm_type == "min_max":
            return min_max_normalize(X, floor=floor)
        if norm_type == "log":
            return lognormalize(X, floor=floor, min_db=min_db)
    return librosa.util.normalize(X, norm=norm_type, axis=1)


def remove_empty_segments(times, labels):
    """Removes empty segments if needed."""
    assert len(times) - 1 == len(labels)
    inters = times_to_intervals(times)
    new_inters = []
    new_labels = []
    for inter, label in zip(inters, labels):
        if inter[0] < inter[1] - 1:
            new_inters.append(inter)
            new_labels.append(label)
    return intervals_to_times(np.asarray(new_inters)), new_labels


def valid_feat_files(file_struct, feat_id, feat_config):

    feat_file = file_struct.get_feat_filename(feat_id)
    if not feat_file.exists():
        return False
    try:
        with open(file_struct.json_file, 'r') as f:
            saved_state = ujson.load(f)
        current_config = vars(getattr(feat_config, feat_id))
    except:
        return False
    return True


def create_json_metadata(audio_file, duration, feat_config):
    if duration is None:
        duration = -1
    out_json = collections.OrderedDict({"metadata": {
        "versions": {"librosa": librosa.__version__,
                     "numpy": np.__version__},
        "timestamp": datetime.datetime.today().strftime(
            "%Y/%m/%d %H:%M:%S")}})
    # Global parameters
    out_json["globals"] = {
        "duration": duration,
        "sample_rate": feat_config.sample_rate,
        "hop_length": feat_config.hop_length,
        "audio_file": str(audio_file.name)
    }
    return out_json
