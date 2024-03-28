# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random

# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH

from typing import Optional
from cog import BasePredictor, Input, Path
import torch

# Model specific imports
import torchaudio
import subprocess
import numpy as np

from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.solvers.compression import CompressionSolver
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
)
from audiocraft.data.audio import audio_write

from audiocraft.models.builders import get_lm_model, get_compression_model, get_wrapped_compression_model
from omegaconf import OmegaConf

import subprocess

def _delete_param(cfg, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)

def load_ckpt(path, device):
    loaded = torch.hub.load_state_dict_from_url(str(path))
    cfg = OmegaConf.create(loaded['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_chord.cache_path')
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    lm = get_lm_model(loaded['xp.cfg'])
    lm.load_state_dict(loaded['model']) 
    lm.eval()
    lm.cfg = cfg
    compression_model = CompressionSolver.model_from_checkpoint(cfg.compression_model_checkpoint, device=device)
    return MusicGen(f"{os.getenv('COG_USERNAME')}/musicgen-finetuned", compression_model, lm)

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mbd = MultiBandDiffusion.get_mbd_musicgen()

        if str(weights) == "weights":
            weights = None

        self.tuned_weights = None

        # if weights is not None:
        #     print("Fine-tuned model weights loaded!")
        #     self.model = load_ckpt(weights, self.device)
        # else:
        self.model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-stereo-melody",
        )
            # self.melody_model = self._load_model(
            #     model_path=MODEL_PATH,
            #     cls=MusicGen,
            #     model_id="facebook/musicgen-melody",
            # )

            # self.large_model = self._load_model(
            #     model_path=MODEL_PATH,
            #     cls=MusicGen,
            #     model_id="facebook/musicgen-large",
            # )

            # self.medium_model = self._load_model(
            #     model_path=MODEL_PATH,
            #     cls=MusicGen,
            #     model_id="facebook/musicgen-medium",
            # )

            # self.small_model = self._load_model(
            #     model_path=MODEL_PATH,
            #     cls=MusicGen,
            #     model_id="facebook/musicgen-small",
            # )

    def _load_model(
        self,
        model_path: str,
        cls: Optional[any] = None,
        load_args: Optional[dict] = {},
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> MusicGen:

        if device is None:
            device = self.device

        compression_model = load_compression_model(
            model_id, device=device, cache_dir=model_path
        )
        lm = load_lm_model(model_id, device=device, cache_dir=model_path)

        return MusicGen(model_id, compression_model, lm)

    def predict(
        self,
        # model_version: str = Input(
        #     description="Model to use for generation. If the model is fine-tuned from MusicGen, then only `finetuned` will work in the newly created fine-tuned model repository.",
        #     default="medium",
        #     choices=["melody", "small", "medium", "large", "encode-decode", "finetuned"],
        # ),
        prompt: str = Input(
            description="A description of the music you want to generate.", default=None
        ),
        input_audio: Path = Input(
            description="An audio file that will influence the generated music. If `continuation` is `True`, the generated music will be a continuation of the audio file. Otherwise, the generated music will mimic the audio file's melody.",
            default=None,
        ),
        duration: int = Input(
            description="Duration of the generated audio in seconds.", default=8
        ),
        continuation: bool = Input(
            description="If `True`, generated music will continue `melody`. Otherwise, generated music will mimic `audio_input`'s melody.",
            default=False,
        ),
        continuation_start: int = Input(
            description="Start time of the audio file to use for continuation.",
            default=0,
            ge=0,
        ),
        continuation_end: int = Input(
            description="End time of the audio file to use for continuation. If -1 or None, will default to the end of the audio clip.",
            default=None,
            ge=0,
        ),
        multi_band_diffusion: bool = Input(
            description="If `True`, the EnCodec tokens will be decoded with MultiBand Diffusion. Only works with non-stereo models.",
            default=False,
        ),
        normalization_strategy: str = Input(
            description="Strategy for normalizing audio.",
            default="loudness",
            choices=["loudness", "clip", "peak", "rms"],
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
        seed: int = Input(
            description="Seed for random number generator. If None or -1, a random seed will be used.",
            default=None,
        ),
        replicate_weights: str = Input(
            description="Replicate MusicGen weights to use. Leave blank to use the default weights.",
            default=None,
        ),
    ) -> Path:

        if prompt is None and input_audio is None:
            raise ValueError("Must provide either prompt or input_audio")
        if continuation and not input_audio:
            raise ValueError("Must provide `input_audio` if continuation is `True`.")
        # if model_version == "large" and input_audio and not continuation:
        #     raise ValueError(
        #         "Large model does not support melody input. Set `model_version='melody'` to condition on audio input."
        #     )
        # elif model_version == "medium" and input_audio and not continuation:
        #     raise ValueError(
        #         "Medium model does not support melody input. Set `model_version='melody'` to condition on audio input."
        #     )
        # elif model_version == "small" and input_audio and not continuation:
        #     raise ValueError(
        #         "Small model does not support melody input. Set `model_version='melody'` to condition on audio input."
        #     )
        # elif model_version == "finetuned":
        #     try:
        #         self.my_model
        #     except:
        #         raise NameError(
        #             "There is no fine-tuned 'weight' file found. Is the model page you are running with is created from additional training process?"
        #         )
        # elif model_version != "finetuned":
        #     try:
        #         self.my_model
        #         raise NameError(
        # #             "You must set `model_version` value as `finetuned`, when the model is fine-tuned from MusicGen."
        # #         )
        #     except:
        #         pass

        # if model_version == "melody":
        #     model = self.melody_model
        # elif model_version == "large":
        #     model = self.large_model
        # elif model_version == "medium":
        #     model = self.medium_model
        # elif model_version == "small":
        #     model = self.small_model
        # elif model_version == "finetuned":
        #     model = self.my_model

        if replicate_weights and replicate_weights != self.tuned_weights:
            self.model = load_ckpt(replicate_weights, self.device)
            print("Fine-tuned model weights hot-swapped!")
            self.tuned_weights = replicate_weights

        if multi_band_diffusion and int(self.model.lm.cfg.transformer_lm.n_q) == 8:
            raise ValueError("Multi-band Diffusion only works with non-stereo models.")

        model = self.model

        set_generation_params = lambda duration: model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )
        
        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")

        '''
        if duration > 30:
            import math

            encodec_rate = 50
            sub_duration=15
            overlap = 30 - sub_duration
            wavs = []
            wav_sr = model.sample_rate
            set_generation_params(30)
            total_step = math.ceil((duration-overlap)/sub_duration)

            if input_audio is None: # Case 1
                print(f"Step 1/{total_step}")
                wav, tokens = model.generate([prompt], progress=True, return_tokens=True)
                if multi_band_diffusion:
                    wav = self.mbd.tokens_to_wav(tokens)
                wavs.append(wav.detach().cpu())
                for i in range((duration - overlap) // sub_duration - 1):
                    print(f"Step {i+2}/{total_step}")
                    wav, tokens= model.generate_continuation_with_audio_token(
                    prompt=tokens[...,sub_duration*encodec_rate:],
                    descriptions=[prompt],
                    progress=True,
                    return_tokens=True
                    )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
                if (duration - overlap) % sub_duration != 0:
                    print(f"Step {total_step}/{total_step}")
                    set_generation_params(overlap + ((duration - overlap) % sub_duration))
                    wav, tokens = model.generate_continuation_with_audio_token(
                        prompt=tokens[...,sub_duration*encodec_rate:],
                        descriptions=[prompt],
                        progress=True,
                        return_tokens=True
                    )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
            elif input_audio is not None and continuation:
                input_audio, sr = torchaudio.load(input_audio)
                input_audio = input_audio[None] if input_audio.dim() == 2 else input_audio

                continuation_start = 0 if not continuation_start else continuation_start
                if continuation_end is None or continuation_end == -1:
                    continuation_end = input_audio.shape[2] / sr

                if continuation_start > continuation_end:
                    raise ValueError(
                        "`continuation_start` must be less than or equal to `continuation_end`"
                    )
                if continuation_end - continuation_start >= 30:
                    raise ValueError(
                        f"input_audio duration({continuation_end - continuation_start}) must be < 30 seconds. Use `continuation_start` and `continuation_end` to trim the input_audio out."
                    )

                input_audio = input_audio[
                    ..., int(sr * continuation_start) : int(sr * continuation_end)
                ]

                print(f"Step 1/{total_step}")
                wav, tokens  = model.generate_continuation(
                            prompt=input_audio,
                            prompt_sample_rate=sr,
                            descriptions=[prompt],
                            progress=True,
                            return_tokens=True
                        )           
                if multi_band_diffusion:
                    wav = self.mbd.tokens_to_wav(tokens)
                wavs.append(wav.detach().cpu())
                for i in range((duration - overlap) // sub_duration - 1):
                    print(f"Step {i+2}/{total_step}")
                    wav, tokens= model.generate_continuation_with_audio_token(
                    prompt=tokens[...,sub_duration*encodec_rate:],
                    descriptions=[prompt],
                    progress=True,
                    return_tokens=True
                    )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
                if (duration - overlap) % sub_duration != 0:
                    print(f"Step {total_step}/{total_step}")
                    set_generation_params(overlap + ((duration - overlap) % sub_duration))
                    wav, tokens = model.generate_continuation_with_audio_token(
                        prompt=tokens[...,sub_duration*encodec_rate:],
                        descriptions=[prompt],
                        progress=True,
                        return_tokens=True
                    )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
            else :
                input_audio, sr = torchaudio.load(input_audio)
                input_audio = input_audio[None] if input_audio.dim() == 2 else input_audio

                continuation_start = 0 if not continuation_start else continuation_start
                
                if continuation_end is None or continuation_end == -1:
                    continuation_end = input_audio.shape[-1] / sr

                if continuation_start > continuation_end:
                    raise ValueError(
                        "`continuation_start` must be less than or equal to `continuation_end`"
                    )

                input_audio = input_audio[
                    ..., int(sr * continuation_start) : int(sr * continuation_end)
                ]

                if input_audio.shape[-1]/sr > duration:
                    wav, tokens = model.generate_with_chroma(['the intro of ' + prompt], input_audio[...,:30*sr], sr, progress=True, return_tokens=True)
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
                    for i in range(int((duration - overlap) // sub_duration) - 1):
                        wav, tokens = model.generate_continuation_with_audio_tokens_and_audio_chroma(
                        prompt=tokens[...,sub_duration*encodec_rate:],
                        melody_wavs = input_audio[...,sub_duration*(i+1)*sr:(sub_duration*(i+1)+30)*sr],
                        melody_sample_rate=sr,
                        descriptions=['chorus of ' + prompt],
                        progress=True,
                        return_tokens=True
                        )
                        if multi_band_diffusion:
                            wav = self.mbd.tokens_to_wav(tokens)
                        wavs.append(wav.detach().cpu())
                    if int(duration - overlap) % sub_duration != 0:
                        set_generation_params(overlap + ((duration - overlap) % sub_duration))
                        wav, tokens = model.generate_continuation_with_audio_tokens_and_audio_chroma(
                            prompt=tokens[...,sub_duration*encodec_rate:],
                            melody_wavs = input_audio[...,sub_duration*(len(wavs))*sr:],
                            melody_sample_rate=sr,
                            descriptions=['the outro of ' + prompt],
                            progress=True,
                            return_tokens=True
                        )
                        if multi_band_diffusion:
                            wav = self.mbd.tokens_to_wav(tokens)
                        wavs.append(wav.detach().cpu())
                else:
                    raise ValueError("Infinite generation(duration > 30) is available with melody condition `input_audio` longer than `duration`.")

            wav = wavs[0][...,:sub_duration*wav_sr]
            for i in range(len(wavs)-1):
                if i == len(wavs)-2:
                    wav = torch.concat([wav,wavs[i+1]],dim=-1)
                else:
                    wav = torch.concat([wav,wavs[i+1][...,:sub_duration*wav_sr]],dim=-1)

            wav = wav.cpu()
        
        else:
        '''
        if not input_audio:
            set_generation_params(duration)
            wav, tokens = model.generate([prompt], progress=True, return_tokens=True)

        # elif model_version == "encode-decode":
        #     encoded_audio = self._preprocess_audio(input_audio, model)
        #     set_generation_params(duration)
        #     wav = model.compression_model.decode(encoded_audio).squeeze(0)

        else:
            input_audio, sr = torchaudio.load(input_audio)
            input_audio = input_audio[None] if input_audio.dim() == 2 else input_audio

            continuation_start = 0 if not continuation_start else continuation_start
            if continuation_end is None or continuation_end == -1:
                continuation_end = input_audio.shape[2] / sr

            if continuation_start > continuation_end:
                raise ValueError(
                    "`continuation_start` must be less than or equal to `continuation_end`"
                )

            input_audio_wavform = input_audio[
                ..., int(sr * continuation_start) : int(sr * continuation_end)
            ]
            input_audio_duration = input_audio_wavform.shape[-1] / sr

            if continuation:
                set_generation_params(duration)
                wav, tokens = model.generate_continuation(
                    prompt=input_audio_wavform,
                    prompt_sample_rate=sr,
                    descriptions=[prompt],
                    progress=True,
                    return_tokens=True,
                )

            else:
                set_generation_params(duration)
                wav, tokens = model.generate_with_chroma(
                    [prompt], input_audio_wavform, sr, progress=True, return_tokens=True
                )
        if multi_band_diffusion:
            wav = self.mbd.tokens_to_wav(tokens)

        audio_write(
            "out",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
        )
        wav_path = "out.wav"

        if output_format == "mp3":
            mp3_path = "out.mp3"
            if Path(mp3_path).exists():
                os.remove(mp3_path)
            subprocess.call(["ffmpeg", "-i", wav_path, mp3_path])
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path

        return Path(path)

    # def _preprocess_audio(
    #     audio_path, model: MusicGen, duration: tp.Optional[int] = None
    # ):

    #     wav, sr = torchaudio.load(audio_path)
    #     wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    #     wav = wav.mean(dim=0, keepdim=True)

    #     # Calculate duration in seconds if not provided
    #     if duration is None:
    #         duration = wav.shape[1] / model.sample_rate

    #     # Check if duration is more than 30 seconds
    #     if duration > 30:
    #         raise ValueError("Duration cannot be more than 30 seconds")

    #     end_sample = int(model.sample_rate * duration)
    #     wav = wav[:, :end_sample]

    #     assert wav.shape[0] == 1
    #     assert wav.shape[1] == model.sample_rate * duration

    #     wav = wav.cuda()
    #     wav = wav.unsqueeze(1)

    #     with torch.no_grad():
    #         gen_audio = model.compression_model.encode(wav)

    #     codes, scale = gen_audio

    #     assert scale is None

    #     return codes


# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
