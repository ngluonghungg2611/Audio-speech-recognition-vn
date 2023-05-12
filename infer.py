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
import wave
import threading
#--------------------------------------------------------
# global frames = []
CHUNK = 1024 #Kich thuoc moi chunk du lieu am thanh
FORMAT = pyaudio.paInt16 #Dinh dang am thanh
CHANNELS = 1 #So kenh am thanh (mono = 1, stereo = 2)
RATE = 16000
WAVE_OUTPUT_FILENAME = "test"
p = pyaudio.PyAudio()
frames = [] 
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

#-----------------------------------------------------------
origins = ["*"]
with open("s2t_rec/config.yaml") as f:
    cfg = yaml.safe_load(f)
# ja_s2t = Vosk_S2T()
vn_s2t = VN_S2T(cfg)

if not os.path.exists("./save_chunks"):
    os.makedirs("./save_chunks")
    
#---------------------
# Vòng lặp để biểu diễn dữ liệu âm thanh theo thời gian thực
def read_audio():
    global frames
    global audio_file
    global time_audio
    strart_time = time.time()
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
        if time_audio == 10:
            strart_time = time.time()
            out_put_path = os.path.join(output_dir, WAVE_OUTPUT_FILENAME+str(output_id) + ".wav")
            wf = wave.open(out_put_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            # text = vn_s2t.speech2text(out_put_path)
            # list_text += text
            # print(list_text)
            audio_file = out_put_path
            output_id += 1
        
        
def pred_audio():
    global audio_file
    global time_audio
    list_text = []
    print("===Save audio and predict===")
    print("==========================================================")
    if time_audio == 10:
        text = vn_s2t.speech2text(audio_file)
        list_text += text
        print(list_text)

thread_read_audio = threading.Thread(target=read_audio)
thread_pred_audio = threading.Thread(target=pred_audio)

thread_read_audio.start()
thread_read_audio.start()

thread_read_audio.join()
thread_pred_audio.join()
