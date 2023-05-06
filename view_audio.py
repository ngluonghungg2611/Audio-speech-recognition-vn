import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as disp
import keyboard
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
WAVE_OUTPUT_FILENAME = "test.wav"

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

# Vòng lặp để biểu diễn dữ liệu âm thanh theo thời gian thực
frames = []
while True:
    # Đọc chunk dữ liệu âm thanh từ luồng thu âm
    data = stream.read(CHUNK)
    frames.append(data)
    # Chuyển đổi dữ liệu âm thanh thành mảng numpy
    
    audio_data = np.frombuffer(data, dtype=np.int16)
    # Biểu diễn dữ liệu âm thanh
    plt.plot(audio_data)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Real-time Audio Waveform')
    plt.ylim(-32768, 32767)  # Đặt giới hạn đồ thị theo giá trị âm thanh
    plt.pause(0.01)
    plt.clf()  # Xóa đồ thị trước khi vẽ đồ thị mới
    # if keyboard.is_pressed('q'):
    #     plt.close()  
    #     break
    # if plt.waitforbuttonpress():
    #     break
    if plt.waitforbuttonpress(timeout=0.001):
        key = plt.gcf().canvas.key_press_event
        if key != '':  # kiểm tra xem phím được nhấn có phải là phím bất kỳ không
            break
# plt.show()
# Dừng luồng thu âm và đóng kết nối
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio_data.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

stream.stop_stream()
stream.close()
p.terminate()
