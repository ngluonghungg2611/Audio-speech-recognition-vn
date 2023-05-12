import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as disp
import keyboard
from scipy.signal import lfilter
import os
import shutil
import datetime
# import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import uvicorn
from pathlib import Path
from tempfile import NamedTemporaryFile
# from s2t_rec.s2t_ja import Vosk_S2T
from s2t_rec.s2t_vi import VN_S2T
import yaml
import time
# import torch
# import torchaudio
# from denoiser import pretrained
# from denoiser.dsp import convert_audio
# from pydub import AudioSegment
import wave
CHUNK = 1024  # Kích thước mỗi chunk dữ liệu âm thanh
FORMAT = pyaudio.paInt16  # Định dạng âm thanh
CHANNELS = 1  # Số kênh âm thanh (mono = 1, stereo = 2)
RATE = 16000  # Tốc độ lấy mẫu âm thanh
WAVE_OUTPUT_FILENAME = "test"

p = pyaudio.PyAudio()


#----
# device = "cpu"
# model_denoise = pretrained.dns64().to(device)
#----
# Mở luồng thu âm
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

#---------------------
origins = ["*"]
with open("s2t_rec/config.yaml") as f:
    cfg = yaml.safe_load(f)
# ja_s2t = Vosk_S2T()
vn_s2t = VN_S2T(cfg)

if not os.path.exists("./save_chunks"):
    os.makedirs("./save_chunks")

#---------------------
# Vòng lặp để biểu diễn dữ liệu âm thanh theo thời gian thực
strart_time = time.time()
frames = []
output_id = 0
len_output = len(os.listdir("./save_chunks"))
list_text = ""
output_dir = os.path.join("./save_chunks", str(len_output))
os.makedirs(output_dir)
while True:
    # Đọc chunk dữ liệu âm thanh từ luồng thu âm
    data = stream.read(CHUNK)
    frames.append(data)
    end_time = time.time()
    time_audio = int(end_time - strart_time)
    # print(time_audio)
    # Chuyển đổi dữ liệu âm thanh thành mảng numpy
    audio_data = np.frombuffer(data, dtype=np.int16)
    # Biểu diễn dữ liệu âm thanh
    plt.plot(audio_data)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Real-time Audio Waveform')
    plt.ylim(-32768, 32767)  # Đặt giới hạn đồ thị theo giá trị âm thanh
    plt.pause(0.001)
    plt.clf()  # Xóa đồ thị trước khi vẽ đồ thị mới
    # if keyboard.is_pressed('q'):
    #     plt.close()   
    #     break
    # if plt.waitforbuttonpress():
    #     break
    print(time_audio)
    print("=====")
    if time_audio == 10:
        print("===Save audio and predict===")
        strart_time = time.time()
        out_put_path = os.path.join(output_dir, WAVE_OUTPUT_FILENAME+str(output_id) + ".wav")
        wf = wave.open(out_put_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        text = vn_s2t.speech2text(out_put_path)
        list_text += text
        print(list_text)
        
        output_id += 1
        
    
    # if plt.waitforbuttonpress(timeout=0.001):
    #     key = plt.gcf().canvas.key_press_event
    #     if key != '':  # kiểm tra xem phím được nhấn có phải là phím bất kỳ không
    #         break
# plt.show()
# Dừng luồng thu âm và đóng kết nối

stream.stop_stream()
stream.close()
p.terminate()
