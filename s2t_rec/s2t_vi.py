from s2t_rec.punc_restore_vn.inference import Inference

from pydub import AudioSegment, silence
import torch
import soundfile as sf
import subprocess

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time
import os
import re
from s2t_rec.utils.preprocess_speech import Denoiser


class VN_S2T:
    def __init__(self, cfg):
        self.device = cfg["device_vi"]
        self.processor = Wav2Vec2Processor.from_pretrained(cfg["wav2vec_vi"])
        self.model = Wav2Vec2ForCTC.from_pretrained(cfg["wav2vec_vi"])

        self.model.to(self.device)
        self.punc = Inference(
            cfg["bert_blsmt_punc_vi"],
            device=cfg["device_vi"],
            punc_path=cfg["punc_vocab"],
            tokenizer_pretrain=cfg["bert_model_vi"],
        )

    # Preprocessing the datasets.
    def map_to_array(self, batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch

    def predict(self, path):
        """recognize speech VietNamese for small audio (10-15s)

        Args:
            path (string): path of audio file (just ".mp3" or ".wav")

        Returns:
            string: text of speech
        """

        path = str(path)
        if ".mp3" in path:
            subprocess.call(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-acodec",
                    "pcm_s16le",
                    path.replace(".mp3", ".wav"),
                ]
            )
            path = path.replace(".mp3", ".wav")

        ds = self.map_to_array({"file": path})

        inputs = self.processor(ds["speech"], return_tensors="pt", padding="longest")

        inputs = inputs.input_values.to(self.device)

        logits = self.model(inputs).logits
        pred = torch.argmax(logits, dim=-1)

        pred = self.processor.batch_decode(pred)

        return pred[0]

    def speech2text(self, audio_path, split_time=10 * 1000):
        """process large audio

        Args:
            audio_path (string): path of audio file (just ".mp3" or ".wav")
            split_time: time to split audio (s)

        Returns:
            string: total text of speech
        """
        audio_path = str(audio_path)
        t0 = time.time()
        if ".mp3" in audio_path:
            subprocess.call(
                    "ffmpeg",
                [
                    "-y",
                    "-i",
                    audio_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-acodec",
                    "pcm_s16le",
                    audio_path.replace(".mp3", ".wav"),
                ]
            )
            audio_path = audio_path.replace(".mp3", ".wav")
            sound = Denoiser(audio_path=audio_path)
        elif ".wav" in audio_path:
            sound = Denoiser(audio_path=audio_path)
        else:
            return {"error": "Hãy upload đúng định dạng file mp3 hoặc wav!"}

        begin = 0
        i = 0

        results = ""

        while True:
            """
            split audio 10s -> detect silence to process large audio
            """
            if begin + (split_time) > len(sound):
                sub_sound = sound[begin : len(sound)]
                save_path = f"./s2t_rec/audio_chunks/{i}_split.mp3"
                sub_sound.export(save_path, bitrate="192k", format="mp3")
                # predict speech to text
                pred_ = self.predict(save_path)
                results += pred_ + " "
                os.remove(save_path)
                break

            sub_sound = sound[begin : begin + (split_time)]

            silences = silence.detect_silence(
                sub_sound, min_silence_len=500, silence_thresh=-6
            )

            if len(silences) == 1:
                end = begin + (split_time)
                next_begin = end
            else:
                end = begin + silences[-2][1]
                next_begin = begin + silences[-1][0]

            sub_sound = sound[begin:end]
            # Create 0.5 seconds silence chunk
            chunk_silent = AudioSegment.silent(duration=10)

            # add 0.5 sec silence to beginning and
            # end of audio chunk. This is done so that
            # it doesn't seem abruptly sliced.
            sub_sound = chunk_silent + sub_sound + chunk_silent

            # export audio to file
            save_path = f"./s2t_rec/audio_chunks/{i}_split.mp3"
            sub_sound.export(save_path, bitrate="192k", format="mp3")

            # predict speech to text
            pred_ = self.predict(save_path)
            results += pred_ + " "

            os.remove(save_path)

            begin = next_begin
            i += 1

        results = self.punc.punc(results)
        # clean result of restore punctuation
        results = (
            results.replace("@@", "")
            .replace(",,", ",")
            .replace("..", ".")
            .replace(" ,", ",")
            .replace(" .", ".")
            .replace(" ?", "?")
        )
        # uppercasing letters after '.', '!' and '?'
        punc_filter = re.compile("([.!?]\s*)")
        split_with_punctuation = punc_filter.split(results)
        results = "".join([i.capitalize() for i in split_with_punctuation])

        print(f"Time s2t process for {len(sound)/3600}s: {time.time()-t0}")
        return results
