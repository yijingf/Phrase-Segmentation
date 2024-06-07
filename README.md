

# PhraseSegmentation

This repository is an adaption from a PyTorch [implementation](https://github.com/morgan76/Triplet_Mining) of the paper [A Repetition-based Triplet Mining Approach for Music Segmentation](https://hal.science/hal-04202766/) 
presented at ISMIR 2023.

The overall format based on the
[MSAF](https://ismir2015.ismir.net/LBD/LBD30.pdf) package. 

We use beat from score instead of beat esitimated from audio, and change the scripts accordingly in `midi_feat.py`, `features.py`, `input_output.py`. 


## Table of Contents
0. [Usage](#usage)
0. [Requirements](#requirements)
0. [Citing](#citing)
0. [Contact](#contact)

## Usage

The dataset format should follow:
```
data/
├── info                    # note event files (.json)
├── midi                    # midi files rendered from note event files (.mid)
├── features                # feature files (.npy)
└── references              # references files (.jams), optional
```

Common audio features can be extracted and aligned with. 
```
python midi_feat.py --ds_path ./data
```

The network can be trained with:
```
python trainer.py --ds_path ./data --feat_id mel --model_name "model_20k" --n_epoch 100
```

To segment tracks and save deep embeddings:
```
python segment.py --ds_path ./data --model_name "model_20k"
```

To evaluate the performance

## Requirements
```
conda env create -f environment.yml
```


## Citing
```
@inproceedings{buisson2023repetition,
  title={A Repetition-based Triplet Mining Approach for Music Segmentation},
  author={Buisson, Morgan and Mcfee, Brian and Essid, Slim and Crayencour, Helene-Camille},
  booktitle={International Society for Music Information Retrieval (ISMIR)},
  year={2023}
}
```