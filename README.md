# spivak: <ins>sp</ins>orts <ins>i</ins>ndeed <ins>v</ins>ideo <ins>a</ins>nalysis <ins>k</ins>it
> A toolkit for automatic analysis of sports videos.

## Updates

- [Sept 2024]: we have released pretrained models for action spotting on the SoccerNet dataset.
Please see [Reproducing-results-from-the-SoccerNet-action-spotting-challenge-2022.md](Reproducing-results-from-the-SoccerNet-action-spotting-challenge-2022.md).

## Background

This package implements methods for action spotting and camera
shot segmentation on the
[SoccerNet dataset](https://www.soccer-net.org/). In addition,
it provides a set of tools for visualizing results, and for
visualizing various metrics that are computed against the ground
truth annotations.

Most of the code in this repository deals with implementing the
following two papers, which focus on the task of action spotting
on the SoccerNet dataset.
- [Temporally Precise Action Spotting in Soccer Videos Using
Dense Detection Anchors](https://arxiv.org/abs/2205.10450).
In ICIP, 2022.
- [Action Spotting using Dense Detection Anchors Revisited:
Submission to the SoccerNet Challenge 2022](https://arxiv.org/abs/2206.07846).
arXiv preprint, 2022.

The action spotting method that is implemented here
came in first place in the SoccerNet Challenge 2022. You can
read more about the 2022 challenge and results in the following paper:
- [SoccerNet 2022 Challenges Results](https://arxiv.org/abs/2210.02365).
In MMSports, 2022.

## Setup

### Requirements

Our models depend on TensorFlow, though this package also includes
some evaluation and visualization code which does not. We've currently
only tested our code using TensorFlow 2.7.0, thus the
corresponding version is currently specified in the
[setup.py](setup.py) file. Certain visualization
scripts depend on ffmpeg via PyAV (`av` pip package). The rest of
the dependencies are specified in [setup.py](setup.py) and can
likely be directly installed using pip as described below.

For the heavier sets of features, our standard flow assumes that a good
amount of CPU RAM is available. We recommend having 256GB or more
in order to make things simpler. Our input data pipeline is
responsible for consuming most of the CPU memory and can be tweaked to
consume less at the cost of speed. An example of how to do this is
presented in
[one of our guides](Reproducing-results-from-the-SoccerNet-action-spotting-challenge-2022.md#low-memory-setup).
The code implementing the input data pipeline is in
[tf_dataset.py](spivak/models/tf_dataset.py), and is based on
[tf.data](https://www.tensorflow.org/guide/data).

### Install

Run pip install in order to do a local development install, as follows.
After successful installation, you should see the message
`Successfully installed spivak`.

```bash
BASE_CODE_DIR="YOUR_BASE_CODE_DIR"  # Wherever the spivak repo is located.
cd $BASE_CODE_DIR/spivak  # The root folder, which contains setup.py.
pip install -e .
```

If you have `ffmpeg` installed, you can also run the video visualization
scripts from our package. In that case, you can run the installation with
the following command, which will also install the PyAV package:

```bash
pip install -e .[av]
```

### Get the SoccerNet data

You will most likely want to download the following set of
precomputed features and labels using SoccerNet's pip package.
Please see detailed instructions at <https://www.soccer-net.org/data>.
- ResNet features
(filenames used for downloading: `["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"]`).
- ResNet features projected using PCA
(filenames used for downloading: `["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"]`).
- Baidu features
(these were also denoted _Combination_ in our papers; filenames
used for downloading:
`["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"]`).
- Action spotting labels
(filename used for downloading: `["Labels-v2.json"]`).
- Camera shot segmentation labels
(filename used for downloading: `["Labels-cameras.json"]`).

If you would like to use our video-specific visualization functionality,
you will also need to get the
[SoccerNet videos](https://www.soccer-net.org/data#h.ov9k48lcih5g).
The low-resolution version of the videos should be enough for
visualization purposes.

### Set up some folders

In order to follow our guides, please create
folders to store your models and results, as follows.

```bash
MODELS_DIR="YOUR_MODELS_DIR"
mkdir -p $MODELS_DIR
RESULTS_DIR="YOUR_RESULTS_DIR"
mkdir -p $RESULTS_DIR
REPORTS_DIR="YOUR_REPORTS_DIR"
mkdir -p $REPORTS_DIR
VISUALIZATIONS_DIR="YOUR_VISUALIZATIONS_DIR"
mkdir -p $VISUALIZATIONS_DIR
# We recommend setting FEATURES_DIR to "data/features" as below, since many
# commands in our guides use "data/features" directly.
FEATURES_DIR="data/features"
mkdir -p $FEATURES_DIR
```

Please also create the symbolic links described below, so
that you can easily access the downloaded SoccerNet data. The symbolic links
will point from the [data/](data) folder to the folders containing the actual
downloaded data.

```bash
cd data/  # This folder will initially just contain the splits/ folder.
ln -s YOUR_LABELS_FOLDER  labels  # For the Labels-v2.json and/or the Labels-cameras.json files.
ln -s YOUR_FEATURES_RESNET_FOLDER  features/resnet  # For the ResNet-based features.
ln -s YOUR_FEATURES_BAIDU_FOLDER  features/baidu  # For the Baidu Combination features.
ln -s YOUR_VIDEOS_224P_FOLDER  videos_224p  # For the low-resolution videos.
```

## Action spotting usage

After completing the setup steps above, please see
[Action-spotting-usage.md](Action-spotting-usage.md) for action spotting
usage instructions. Additionally, to download our pretrained action
spotting models and to reproduce results from our experiments, please see
[Reproducing-results-from-the-SoccerNet-action-spotting-challenge-2022.md](Reproducing-results-from-the-SoccerNet-action-spotting-challenge-2022.md).

## Citations

If you found our models and code useful, please consider citing our works:

```
@inproceedings{soares2022temporally,
  author={Soares, Jo{\~a}o~V.~B. and Shah, Avijit and Biswas, Topojoy},
  booktitle={International Conference on Image Processing (ICIP)},
  title={Temporally Precise Action Spotting in Soccer Videos Using Dense Detection Anchors},
  year={2022},
  pages={2796-2800},
  doi={10.1109/ICIP46576.2022.9897256}
}

@article{soares2022action,
  title={Action Spotting using Dense Detection Anchors Revisited: Submission to the {SoccerNet} {Challenge} 2022},
  author={Soares, Jo{\~a}o~V.~B. and Shah, Avijit},
  journal={arXiv preprint arXiv:2206.07846},
  year={2022}
}
```

## Contribute

Please refer to [the Contributing.md file](Contributing.md) for information about how to
get involved. We welcome  issues, questions, and pull requests.

Please be aware that we (the maintainers) are currently busy with other projects, so it
may take some days before we are able to get back to you. We do not foresee big changes
to this repository going forward.

## Maintainers

- Joao Soares: jvbsoares@yahooinc.com
- Avijit Shah: avijit.shah@yahooinc.com

## License

This project is licensed under the terms of the
[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0.html)
open source license. Please refer to [LICENSE](LICENSE) for
the full terms.

## Acknowledgments

We thank the [SoccerNet team](https://www.soccer-net.org/team) for making their datasets
available and organizing the series of related challenges. We also thank them for
making their code available under open source licenses.
