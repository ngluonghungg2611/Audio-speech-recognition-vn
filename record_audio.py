import pyaudio
import wave
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
#----------- CLI ---------------
parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=int, default=5, help="Import time to record")
args = parser.parse_args()
# thiết lập thông số âm thanh
id_file = len(os.listdir("./audio_test"))
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = args.seconds
WAVE_OUTPUT_FILENAME = "audio_test/{}_audio__{}s.wav".format(str(id_file) ,str(RECORD_SECONDS))

# khởi tạo PyAudio
audio = pyaudio.PyAudio()

# bắt đầu ghi âm
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("Đang ghi âm...")
frames = []

for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS)), desc="Recording..."):
    data = stream.read(CHUNK)
    frames.append(data)


print("Đã ghi âm xong!")
stream.stop_stream()
stream.close()

# plt.plot(frames)
# plt.xlabel("Thời gian (mẫu)")
# plt.ylabel("Amplitude")
# plt.title("Biểu đồ âm thanh")
# plt.show()
# audio.terminate()

# lưu tệp âm thanh
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
# kết thúc ghi âm