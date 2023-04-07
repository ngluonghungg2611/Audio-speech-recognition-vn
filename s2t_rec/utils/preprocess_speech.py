from IPython import display as disp
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from pydub import AudioSegment

device = "cpu"
model_denoise = pretrained.dns64().to(device)


def Denoiser(audio_path=""):
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(
        wav.to(device), sr, model_denoise.sample_rate, model_denoise.chin
    )
    with torch.no_grad():
        denoised = model_denoise(wav[None])[0]
    audio = disp.Audio(denoised.data.cpu().numpy(), rate=model_denoise.sample_rate)
    audio = AudioSegment(
        audio.data, frame_rate=model_denoise.sample_rate, sample_width=2, channels=1
    )
    return audio
