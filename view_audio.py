import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Đọc tệp âm thanh
sample_rate, audio = wavfile.read("output.wav")

# Chuyển đổi dữ liệu âm thanh sang dạng số thực
audio = audio.astype(np.float32)

# Tính toán số mẫu và thời gian tương ứng
num_samples = len(audio)
duration = num_samples / sample_rate

# Tính toán độ phân giải của hình ảnh
resolution = 1000  # Độ phân giải của hình ảnh
samples_per_pixel = int(num_samples / resolution)
horizontal_resolution = samples_per_pixel * resolution
vertical_resolution = 256

# Tính toán dữ liệu của từng pixel
data = np.zeros((vertical_resolution, horizontal_resolution), dtype=np.float32)
for i in range(resolution):
    start_sample = i * samples_per_pixel
    end_sample = start_sample + samples_per_pixel
    pixel_data = audio[start_sample:end_sample]
    pixel_max = np.max(np.abs(pixel_data))
    data[:, i * samples_per_pixel:(i + 1) * samples_per_pixel] = np.abs(pixel_data) / pixel_max

# Hiển thị hình ảnh
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(data, cmap='gray', aspect='auto', interpolation='nearest')
ax.set_xlim(0, horizontal_resolution)
ax.set_ylim(vertical_resolution, 0)
ax.axis('off')
plt.title("Hình ảnh âm thanh")
plt.show()