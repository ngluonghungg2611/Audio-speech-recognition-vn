import pyaudio
import wave
from tqdm import tqdm
import os
import cv2
import threading
# thiết lập thông số âm thanh
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
WAVE_OUTPUT_FILENAME = os.path.join(output_dir, "output_{}.wav".format(len(os.listdir(output_dir))))

class AudioRecorder:
    def __init__(self):
        self.frames = []
        self.recording = False

    def start_recording(self):
        # Khởi tạo đối tượng PyAudio
        self.audio = pyaudio.PyAudio()

        # Bắt đầu ghi âm
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

        self.recording = True

        # Bắt đầu một thread mới để ghi âm và xử lý dữ liệu
        self.thread = threading.Thread(target=self.record)
        self.thread.start()

    def stop_recording(self):
        self.recording = False
        self.thread.join()

        # Dừng ghi âm
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        # Lưu dữ liệu âm thanh vào tệp WAV
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()

    def record(self):
        while self.recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)

            # Xử lý dữ liệu âm thanh tại đây
            # ...

recorder = AudioRecorder()
recorder.start_recording()

# Chạy ghi âm trong 10 giây
import time
time.sleep(15)

recorder.stop_recording()