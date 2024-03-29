import logging
import os
import os.path
import json

from cog import BaseModel, Input, Path
import subprocess as sp
import shutil
from tqdm import tqdm
from tinytag import TinyTag
from pydub import AudioSegment
import librosa
from numba import cuda
import torch
from metadata import genre_labels, mood_theme_classes, instrument_classes
import demucs
from demucs.apply import apply_model
from demucs.audio import convert_audio
import numpy as np
import torchaudio
import moviepy
import audiocraft.data.audio_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
model_files = [
    "genre_discogs400-discogs-effnet-1.pb",
    "discogs-effnet-bs64-1.pb",
    "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "mtg_jamendo_instrument-discogs-effnet-1.pb",
]

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)


class TrainingOutput(BaseModel):
    weights: Path


def filter_predictions(predictions, class_list, threshold):
    predictions_mean = np.mean(predictions, axis=0)
    sorted_indices = np.argsort(predictions_mean)[::-1]
    filtered_indices = [i for i in sorted_indices if predictions_mean[i] > threshold]
    filtered_labels = [class_list[i] for i in filtered_indices]
    filtered_values = [predictions_mean[i] for i in filtered_indices]
    return filtered_labels, filtered_values

def make_comma_separated_unique(tags):
    seen_tags = {tag.lower().strip() for tag in tags if tag is not None and tag is not ""}
    return ", ".join(list(seen_tags))

def get_audio_features(audio_filename):
    audio = MonoLoader(filename=audio_filename, sampleRate=16000, resampleQuality=4)()
    # Load ID3 tags if available
    metadata = TinyTag.get(audio_filename)

    result_dict = {
        "artist": metadata.artist,
        "title": metadata.title,
        "description": (metadata.comment or "") + metadata.extra.get("description", ""),
    }

    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1"
    )
    embeddings = embedding_model(audio)

    # Predicting genres
    genre_model = TensorflowPredict2D(
        graphFilename="genre_discogs400-discogs-effnet-1.pb",
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    predictions = genre_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, genre_labels, threshold=0.15)
    filtered_labels = ", ".join(filtered_labels).replace("---", ", ").split(", ")
    print({"auto.genre": ",".join(filtered_labels)})
    if metadata.genre is not None:
        print({"metadata.genre": metadata.genre})
        filtered_labels = filtered_labels + metadata.genre.split(",")
    result_dict["genres"] = make_comma_separated_unique(filtered_labels)

    # Predicting mood/theme
    mood_model = TensorflowPredict2D(graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb")
    predictions = mood_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, mood_theme_classes, threshold=0.06)
    print({"auto.moods": ",".join(filtered_labels)})
    result_dict["moods"] = make_comma_separated_unique(filtered_labels)

    # Predicting instruments
    instrument_model = TensorflowPredict2D(graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb")
    predictions = instrument_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, instrument_classes, threshold=0.15)
    print({"auto.instruments": ",".join(filtered_labels)})
    result_dict["instruments"] = filtered_labels

    return result_dict

def prepare_data(
    dataset_path: Path,
    target_path: str = "src/train_data",
    one_same_description: str = None,
    meta_path: str = "src/meta",
    auto_labeling: bool = True,
    drop_vocals: bool = True,
    device: str = "cuda",
):

    d_path = Path(target_path)
    d_path.mkdir(exist_ok=True, parents=True)

    # Decompressing file at dataset_path
    ext = str(dataset_path).rsplit(".", 1)[1]
    if ext == "zip":
        sp.call(["unzip", str(dataset_path), "-d", target_path + "/"])
    elif ext == "tar":
        sp.call(["tar", "-xvf", str(dataset_path), "-C", target_path + "/"])
    elif ext == "gz":
        sp.call(["tar", "-xvzf", str(dataset_path), "-C", target_path + "/"])
    elif ext == "tgz":
        sp.call(["tar", "-xzvf", str(dataset_path), "-C", target_path + "/"])
    elif ext in ["wav", "mp3", "flac", "mp4"]:
        shutil.move(str(dataset_path), target_path + "/" + str(dataset_path.name))
    else:
        raise Exception(
            "Not supported compression file type. The file type should be one of 'zip', 'tar', 'tar.gz', 'tgz' types of compression file, or a single 'wav', 'mp3', 'flac', 'mp4' types of audio file."
        )

    # Removing __MACOSX and .DS_Store
    if (Path(target_path) / "__MACOSX").is_dir():
        shutil.rmtree(target_path + "/__MACOSX")
    elif (Path(target_path) / "__MACOSX").is_file():
        os.remove(target_path + "/__MACOSX")
    if (Path(target_path) / ".DS_Store").is_dir():
        shutil.rmtree(target_path + "/.DS_Store")
    elif (Path(target_path) / ".DS_Store").is_file():
        os.remove(target_path + "/.DS_Store")

    # Audio Chunking and Vocal Dropping
    if drop_vocals:
        separator = demucs.pretrained.get_model("htdemucs_ft")
        if device == "cuda":
            separator = separator.cuda()
        else:
            separator = separator.cpu()
    else:
        separator = None

    for filename in tqdm(os.listdir(target_path)):
        if filename.endswith((".mp3", ".wav", ".flac", ".mp4")):
            if filename.endswith((".mp4")):
                video = moviepy.editor.VideoFileClip(os.path.join(target_path, filename))
                fname = filename.rsplit(".", 1)[0] + ".wav"
                video.audio.write_audiofile(os.path.join(target_path, fname))
                print(f"A mp4 file is converted into a wav file : {filename}")
                os.remove(target_path + "/" + filename)
            else:
                fname = filename

            # Chuking audio files into 30sec chunks
            audio = AudioSegment.from_file(target_path + "/" + fname)
            audio = audio.set_frame_rate(44100)  # Resampling to 44100

            if len(audio) > 30000:
                print("Chunking " + fname)

                # Splitting the audio files into 30-second chunks
                for i in range(0, len(audio), 30000):
                    chunk = audio[i : i + 30000]
                    if len(chunk) > 5000:  # Omitting residuals with <5sec duration
                        if drop_vocals and separator is not None:
                            chunk_fname = f"{target_path}/{fname[:-4]}_chunk{i//1000}.wav"
                            print("Separating Vocals from " + chunk_fname)

                            channel_sounds = chunk.split_to_mono()
                            samples = [s.get_array_of_samples() for s in channel_sounds]

                            chunk = np.array(samples).T.astype(np.float32)
                            chunk /= np.iinfo(samples[0].typecode).max
                            chunk = torch.Tensor(chunk).T
                            print(chunk.shape)

                            # Resample for Demucs
                            chunk = convert_audio(chunk, 44100, separator.samplerate, separator.audio_channels)
                            stems = apply_model(separator, chunk[None], device=device, shifts=4)
                            stems = stems[
                                :,
                                [
                                    separator.sources.index("bass"),
                                    separator.sources.index("drums"),
                                    separator.sources.index("other"),
                                ],
                            ]
                            mixed = stems.sum(1)
                            torchaudio.save(chunk_fname, mixed.squeeze(0), separator.samplerate)
                        else:
                            chunk.export(chunk_fname, format="wav")
                os.remove(target_path + "/" + fname)

    max_sample_rate = 0

    # Auto Labeling
    if auto_labeling:
        for model_file in model_files:
            model_url = f"https://essentia.upf.edu/models/{model_file.replace('-', '/')}"
            sp.call(["curl", model_url, "--output", model_file])

        train_len = 0

        os.mkdir(meta_path)
        with open(meta_path + "/data.jsonl", "w", encoding="utf-8") as train_file:
            files = list(d_path.rglob("*.mp3")) + list(d_path.rglob("*.wav")) + list(d_path.rglob("*.flac"))
            if len(files) == 0:
                raise ValueError("No audio file detected. Are you sure the audio file is longer than 5 seconds?")
            for filename in tqdm(files):
                result = get_audio_features(str(filename))

                # Obtaining key and BPM
                y, sr = librosa.load(str(filename))
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = round(tempo)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                key = np.argmax(np.sum(chroma, axis=1))
                key = keys[key]
                length = librosa.get_duration(y=y, sr=sr)

                sr = librosa.get_samplerate(str(filename))
                if sr > max_sample_rate:
                    max_sample_rate = sr

                entry = {
                    "artist": result.get("artist", ""),
                    "title": result.get("title", ""),
                    "description": result.get("description", ""),
                    "key": f"{key}",
                    "sample_rate": sr,
                    "duration": length,
                    "bpm": tempo,
                    "genre": result.get("genres", ""),
                    "instrument": result.get("instruments", ""),
                    "moods": result.get("moods", []),
                    "path": str(filename),
                    "file_extension": "wav",
                    "keywords": "",
                    "name": "",
                }
                with open(str(filename).rsplit(".", 1)[0] + ".json", "w", encoding="utf-8") as file:
                    json.dump(entry, file)
                print(entry)

                train_len += 1
                train_file.write(json.dumps(entry) + "\n")

            device = cuda.get_current_device()
            device.reset()

            filelen = len(files)
    else:
        meta = audiocraft.data.audio_dataset.find_audio_files(
            target_path,
            audiocraft.data.audio_dataset.DEFAULT_EXTS,
            progress=True,
            resolve=False,
            minimal=True,
            workers=10,
        )

        if len(meta) == 0:
            raise ValueError("No audio file detected. Are you sure the audio file is longer than 5 seconds?")

        for m in meta:
            if m.sample_rate > max_sample_rate:
                max_sample_rate = m.sample_rate
            fdict = {
                "key": "",
                "artist": "",
                "sample_rate": m.sample_rate,
                "file_extension": m.path.rsplit(".", 1)[1],
                "description": "",
                "keywords": "",
                "duration": m.duration,
                "bpm": "",
                "genre": "",
                "title": "",
                "name": Path(m.path).name.rsplit(".", 1)[0],
                "instrument": "",
                "moods": [],
            }
            with open(m.path.rsplit(".", 1)[0] + ".json", "w", encoding="utf-8") as file:
                json.dump(fdict, file)
        audiocraft.data.audio_dataset.save_audio_meta(meta_path + "/data.jsonl", meta)
        filelen = len(meta)

    audios = list(d_path.rglob("*.mp3")) + list(d_path.rglob("*.wav"))

    for audio in list(audios):
        with open(str(audio).rsplit(".", 1)[0] + ".json", "r", encoding="utf-8") as jsonf:
            fdict = json.load(jsonf)

        if one_same_description is None:
            if Path(str(audio).rsplit(".", 1)[0] + ".txt").exists():
                with open(str(audio).rsplit(".", 1)[0] + ".txt", "r", encoding="utf-8") as f:
                    line = f.readline()
                fdict["description"] = line
            elif Path(str(audio).rsplit("_chunk", 1)[0] + ".txt").exists():
                with open(str(audio).rsplit("_chunk", 1)[0] + ".txt", "r", encoding="utf-8") as f:
                    line = f.readline()
                fdict["description"] = line
        else:
            fdict["description"] = one_same_description

        with open(str(audio).rsplit(".", 1)[0] + ".json", "w", encoding="utf-8") as file:
            json.dump(fdict, file)

    return max_sample_rate, filelen


def train(
    dataset_path: Path = Input(
        "Path to dataset directory. Input audio files will be chunked into multiple 30 second audio files. Must be one of 'tar', 'tar.gz', 'gz', 'zip' types of compressed file, or a single 'wav', 'mp3', 'flac' file. Audio files must be longer than 5 seconds."
    ),
    auto_labeling: bool = Input(
        description="Creating label data like genre, mood, theme, instrumentation, key, bpm for each track. Using `essentia-tensorflow` for music information retrieval.",
        default=True,
    ),
    drop_vocals: bool = Input(
        description="Dropping the vocal tracks from the audio files in dataset, by separating sources with Demucs.",
        default=True,
    ),
    one_same_description: str = Input(description="A description for all of audio data", default=None),
    model_version: str = Input(
        description="Model version to train.",
        default="stereo-melody",
        choices=[
            "stereo-melody",
            "stereo-melody-large",
            "stereo-large",
            "stereo-small",
            "stereo-medium",
            "melody",
            "small",
            "medium",
        ],
    ),
    epochs: int = Input(
        description="Number of epochs to train for", default=5
    ),  # set to 5 based on this paper: https://ar5iv.labs.arxiv.org/html/2311.09094
    updates_per_epoch: int = Input(description="Number of iterations for one epoch", default=100),
    batch_size: int = Input(
        description="Batch size. Must be multiple of 8(number of gpus), for 8-gpu training.", default=16
    ),
    optimizer: str = Input(description="Type of optimizer.", default="dadam", choices=["dadam", "adamw"]),
    lr: float = Input(description="Learning rate", default=1),
    lr_scheduler: str = Input(
        description="Type of lr_scheduler",
        default="cosine",
        choices=["exponential", "cosine", "polynomial_decay", "inverse_sqrt", "linear_warmup"],
    ),
    warmup: int = Input(description="Warmup of lr_scheduler", default=8),
    cfg_p: float = Input(description="CFG dropout ratio", default=0.3),
) -> TrainingOutput:
    meta_path = "src/meta"
    target_path = "src/train_data"

    out_path = "trained_model.tar"

    # Removing previous training"s leftover
    if os.path.isfile(out_path):
        os.remove(out_path)
    if os.path.isfile("weights"):
        os.remove("weights")
    if os.path.isfile("weight"):
        os.remove("weight")
    if os.path.isdir("weights"):
        shutil.rmtree("weights")
    if os.path.isdir("weight"):
        shutil.rmtree("weight")
    if os.path.isdir(meta_path):
        shutil.rmtree(meta_path)
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)
    if os.path.isdir("models"):
        shutil.rmtree("models")
    if os.path.isdir("tmp"):
        shutil.rmtree("tmp")

    max_sample_rate, len_dataset = prepare_data(
        dataset_path, target_path, one_same_description, meta_path, auto_labeling, drop_vocals, device_to_use
    )

    # max # of GPUs we can get is 8, so we need to set batch size to 8 if the model is large so we don"t OOM
    if "melody" in model_version or "stereo" in model_version or "large" in model_version or "medium" in model_version:
        batch_size = 8
        print(
            f"Batch size is reset to {batch_size}, since complex models can only be trained with 8 with current GPU settings."
        )

    if batch_size % 8 != 0:
        batch_size = batch_size - (batch_size % 8)
        print(f"Batch size is reset to {batch_size}, the multiple of 8(number of gpus).")

    # Setting up dora args
    if "melody" in model_version:
        solver = "musicgen/musicgen_melody_32khz"
        conditioner = "chroma2music"
    else:
        solver = "musicgen/musicgen_base_32khz"
        conditioner = "text2music"

    if "large" in model_version:
        model_scale = "large"
    elif "medium" in model_version:
        model_scale = "medium"
    elif "small" in model_version:
        model_scale = "small"
    else:
        model_scale = "medium"

    continue_from = f"//pretrained/facebook/musicgen-{model_version}"

    args = [
        "run",
        "-d",
        "--",
        f"solver={solver}",
        f"model/lm/model_scale={model_scale}",
        f"continue_from={continue_from}",
        f"conditioner={conditioner}",
    ]
    if "stereo" in model_version:
        args.append(f"codebooks_pattern.delay.delays={[0, 0, 1, 1, 2, 2, 3, 3]}")
        args.append("transformer_lm.n_q=8")
        args.append("interleave_stereo_codebooks.use=True")
        args.append("channels=2")
    args.append(f"datasource.max_sample_rate={max_sample_rate}")
    args.append(f"datasource.train={meta_path}")
    args.append(f"dataset.train.num_samples={len_dataset}")
    args.append(f"optim.epochs={epochs}")
    args.append(f"optim.lr={lr}")
    args.append(f"schedule.lr_scheduler={lr_scheduler}")
    args.append(f"schedule.cosine.warmup={warmup}")
    args.append(f"schedule.polynomial_decay.warmup={warmup}")
    args.append(f"schedule.inverse_sqrt.warmup={warmup}")
    args.append(f"schedule.linear_warmup.warmup={warmup}")
    args.append(f"classifier_free_guidance.training_dropout={cfg_p}")
    if updates_per_epoch is not None:
        args.append(f"logging.log_updates={updates_per_epoch//10 if updates_per_epoch//10 >=1 else 1}")
    else:
        args.append("logging.log_updates=0")
    args.append(f"dataset.batch_size={batch_size}")
    args.append(f"optim.optimizer={optimizer}")

    if updates_per_epoch is None:
        args.append("dataset.train.permutation_on_files=False")
        args.append("optim.updates_per_epoch=1")
    else:
        args.append("dataset.train.permutation_on_files=True")
        args.append(f"optim.updates_per_epoch={updates_per_epoch}")

    print("Starting training!")
    print(
        f"Model Scale: {model_scale} Conditioner: {conditioner} Solver: {solver} Epochs: {epochs} Batch Size: {batch_size}"
    )
    sp.call(["python3", "dora_main.py"] + args)

    checkpoint_dir = None
    for dirpath, _dirnames, filenames in os.walk("tmp"):
        for filename in [f for f in filenames if f == "checkpoint.th"]:
            checkpoint_dir = os.path.join(dirpath, filename)

    if checkpoint_dir is None:
        raise Exception("Training failed - no checkpoint found. Check the logs for more information.")
    loaded = torch.load(checkpoint_dir, map_location=torch.device("cpu"))

    torch.save({"xp.cfg": loaded["xp.cfg"], "model": loaded["model"]}, out_path)

    return TrainingOutput(weights=Path(out_path))
