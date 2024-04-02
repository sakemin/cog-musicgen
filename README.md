# Cog Implementation of MusicGen with Fine-tuner
[![Replicate](https://replicate.com/sakemin/musicgen-fine-tuner/badge)](https://replicate.com/sakemin/musicgen-fine-tuner) 

[MusicGen](https://replicate.com/meta/musicgen) represents a straightforward and manageable model designed for music generation, as described in [this research paper](https://arxiv.org/abs/2306.05284). This model allows users to refine MusicGen using their own datasets.

MusicGen is [a simple and controllable model for music generation](https://arxiv.org/abs/2306.05284).  It is a single stage auto-regressive Transformer model trained over a 32kHz <a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> with 4 codebooks sampled at 50 Hz. Unlike existing methods like [MusicLM](https://arxiv.org/abs/2301.11325), MusicGen doesn't require a self-supervised semantic representation, and it generates all 4 codebooks in one pass. By introducing a small delay between the codebooks, the authors show they can predict them in parallel, thus having only 50 auto-regressive steps per second of audio. They used 20K hours of licensed music to train MusicGen. Specifically, they relied on an internal dataset of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


For more information about this model, see [here](https://github.com/facebookresearch/audiocraft).

You can demo this model or learn how to use it with Replicate's API [here](https://replicate.com/sakemin/musicgen-fine-tuner). 

## Prediction
### Default Model
- In this repository, the default prediction model is configured as the melody model.
- After completing the fine-tuning process from this repository, the trained model weights will be loaded into your own model repository on Replicate.

# Run with Cog

[Cog](https://github.com/replicate/cog) is an open-source tool that packages machine learning models in a standard, production-ready container. 
You can deploy your packaged model to your own infrastructure, or to [Replicate](https://replicate.com/), where users can interact with it via web interface or API.

## Prerequisites 

**Cog.** Follow these [instructions](https://github.com/replicate/cog#install) to install Cog, or just run: 

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

Note, to use Cog, you'll also need an installation of [Docker](https://docs.docker.com/get-docker/).

* **GPU machine.** You'll need a Linux machine with an NVIDIA GPU attached and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. If you don't already have access to a machine with a GPU, check out our [guide to getting a 
GPU machine](https://replicate.com/docs/guides/get-a-gpu-machine).

## Step 1. Clone this repository

```sh
git clone https://github.com/sakemin/cog-musicgen-fine-tuner
```

## Step 2. Run the model

To run the model, you need a local copy of the model's Docker image. You can satisfy this requirement by specifying the image ID in your call to `predict` like:

```
cog predict r8.im/sakemin/musicgen-fine-tuner@sha256:b1ec6490e57013463006e928abc7acd8d623fe3e8321d3092e1231bf006898b1 -i prompt="tense staccato strings. plucked strings. dissonant. scary movie." -i duration=8
```

For more information, see the Cog section [here](https://replicate.com/sakemin/musicgen-fine-tuner/api#run)

Alternatively, you can build the image yourself, either by running `cog build` or by letting `cog predict` trigger the build process implicitly. For example, the following will trigger the build process and then execute prediction: 

```
cog predict -i prompt="tense staccato strings. plucked strings. dissonant. scary movie." -i duration=8
```

Note, the first time you run `cog predict`, model weights and other requisite assets will be downloaded if they're not available locally. This download only needs to be executed once.

# Run on replicate

## Step 1. Ensure that all assets are available locally

If you haven't already, you should ensure that your model runs locally with `cog predict`. This will guarantee that all assets are accessible. E.g., run: 

```
cog predict -i prompt=tense staccato strings. plucked strings. dissonant. scary movie. -i duration=8
```

## Step 2. Create a model on Replicate.

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model. If you want to keep the model private, make sure to specify "private".

## Step 3. Configure the model's hardware

Replicate supports running models on variety of CPU and GPU configurations. For the best performance, you'll want to run this model on an A100 instance.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 4: Push the model to Replicate


Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 1:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)

# Fine-tuning MusicGen

Assuming you have a local environment configured (i.e. you've completed the steps specified under Run with Cog), you can run training with a command like:

```
cog train -i dataset_path=@<path-to-your-data> <additional hyperparameters>
```
## Dataset
### Audio
- Compressed files in formats like .zip, .tar, .gz, and .tgz are compatible for dataset uploads.
- Single audio files with .mp3, .wav, and .flac formats can also be uploaded.
- Audio files within the dataset must exceed 30 seconds in duration.
- **Audio Chunking:** Files surpassing 30 seconds will be divided into multiple 30-second chunks.
- **Vocal Removal:** If `drop_vocals` is set to `True`, the vocal tracks in the audio files will be isolated and removed (Default: `True`).
	- For datasets containing audio without vocals, setting `drop_vocals=False` reduces data preprocessing time and maintains audio file quality.	
### Text Description
- If each audio file requires a distinct description, create a .txt file with a single-line description corresponding to each .mp3 or .wav file (e.g., `01_A_Man_Without_Love.mp3` and `01_A_Man_Without_Love.txt`).
- For a uniform description across all audio files, set the `one_same_description` argument to your desired description. In this case, there's no need for individual .txt files.
- **Auto Labeling:** When `auto_labeling` is set to `True`, labels such as 'genre', 'mood', 'theme', 'instrumentation', 'key', and 'bpm' will be generated and added to each audio file in the dataset (Default: `True`).
	- [Available Tags for Labeling](https://github.com/sakemin/cog-musicgen-fine-tuner/blob/main/metadata.py)
## Train Parameters
### Train Inputs
- `dataset_path`: Path = Input("Path to the dataset directory")
- `one_same_description`: str = Input(description="A description for all audio data", default=None)
- `auto_labeling`: bool = Input(description="Generate labels (genre, mood, theme, etc.) for each track using `essentia-tensorflow` for music information retrieval", default=True)
- `source`: str = Input(description="Apply source separation to input audio ('all' leaves audio uneffected)", default="all")
- `model_version`: str = Input(description="Model version to train", default="stereo-melody", choices=[“melody”, “small”, “medium”, “stereo-melody”, “stereo-small”, “stereo-medium”])
- `lr`: float = Input(description="Learning rate", default=1)
- `epochs`: int = Input(description="Number of epochs to train for", default=10)
- `updates_per_epoch`: int = Input(description="Number of iterations for one epoch", default=100) #If None, iterations per epoch will be set according to dataset/batch size. If a value is provided, the number of iterations per epoch will be set as specified.
- `batch_size`: int = Input(description="Batch size", default=3)
### Default Parameters
- Using `epochs=3`, `updates_per_epoch=100`, and `lr=1`, the fine-tuning process takes approximately 15 minutes.
- For 8 GPU multiprocessing, `batch_size` must be a multiple of 8. Otherwise, `batch_size` will be automatically set to the nearest multiple of 8.
- For the `medium` model, the maximum `batch_size` is `8` with the specified 8 x Nvidia A40 machine setting.

## Example Code with Replicate API
```python
import replicate

training = replicate.trainings.create(
	version="sakemin/musicgen-fine-tuner:b1ec6490e57013463006e928abc7acd8d623fe3e8321d3092e1231bf006898b1",
  input={
    "dataset_path":"https://your/data/path.zip",
    "one_same_description":"description for your dataset music",
    "epochs":3,
    "updates_per_epoch":100,
    "model_version":"medium",
  },
  destination="my-name/my-model"
)

print(training)
```
---
## References
-  Auto-labeling and audio chunking features are based on [lyramakesmusic](https://github.com/lyramakesmusic)'s [Finetune-MusicGen](https://github.com/lyramakesmusic/finetune-musicgen) jupyter notebook.
- The auto-labeling feature utilizes [`effnet-discogs`](https://replicate.com/mtg/effnet-discogs) from [MTG](https://github.com/MTG)'s [`essentia`](https://github.com/MTG/essentia).
- 'key' and 'bpm' values are obtained using `librosa`.
- Vocal dropping is implemented using Meta's [`demucs`](https://github.com/facebookresearch/demucs).
## Licenses
- All code in this repository is licensed under the [Apache License 2.0 license](https://github.com/sakemin/cog-musicgen-fine-tuner/blob/main/LICENSE).
- The code in the [Audiocraft](https://github.com/facebookresearch/audiocraft) repository is released under the MIT license (see [LICENSE file](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE)).
- The weights in the [Audiocraft](https://github.com/facebookresearch/audiocraft) repository are released under the CC-BY-NC 4.0 license (see [LICENSE_weights file](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE_weights)).
