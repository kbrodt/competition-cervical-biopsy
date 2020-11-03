# TissueNet: Detect Lesions in Cervical Biopsies

[TissueNet: Detect Lesions in Cervical Biopsies](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/).

[3rd place](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/leaderboard/)
out of 547 with 0.9339 Weighted Class Score (top 1 -- 0.9475).

### Prerequisites

- [libvips](https://libvips.github.io/libvips/install.html)
- GPU with 32Gb RAM (e.g. Tesla V100)
- [NVIDIA apex](https://github.com/NVIDIA/apex)

```bash
pip install -r requirements.txt
```

### Usage

#### Download

First download the train data from the competition link into `data` folder.

Then you have to download train images

```bash
python ./src/download.py
```

This will download whole dataset (~1Tb) of pyramidal TIF slides.

#### Train

To train the model run

```bash
bash ./train_all.sh
```

On 1 GPU Tesla V100 it will take around 2-3 weeks. If you have more GPUs,
you can [parallelize it](https://www.gnu.org/software/parallel/). Actually,
you do not need to train all ~250 epochs, because the best models on validation
are on ~25 epoch. So you can reduce training to 2-3 days.

To generate torch scrips models run

```bash
bash ./validate_all.sh
```

#### Test

Once the model trained run the following command to do inference on test.
Or you can also download the trained models from [yandex disk](https://yadi.sk/d/85zam_5YNLTcQg),
unzip and run

```bash
python main.py
```

### Approach

#### First observations

#### Summary

#### Highlights

- efficientnet-b0
- 16x downsampled images (`page` 4)
- crop the most `interesting` tiles from large image
- Binary Cross Entropy Loss
- Batch size 8
- AdamW with learning rate `1e-3` or `3e-4`
- CosineAnealing scheduler
- Augmentations on tile and whole image levels: horizontal and vertical flips, rotate on 90
