import wave  # conversion to .wav

import csv  # to output log

import numpy as np
from scipy import signal  # plot spectogram
from scipy.io import wavfile
import matplotlib.pyplot as plt

from pydub import AudioSegment  # for trimming wav

from pathlib import Path


# Change cutoff if needed, currently- 12 kHz
def high_pass_filter(data, sample_rate, cutoff=12000):
    sos = signal.butter(10, cutoff, 'hp', fs=sample_rate, output='sos')
    filtered = signal.sosfilt(sos, data)
    return filtered


def trim_wav(input, start, stop):
    audio = AudioSegment.from_wav(input)
    end = len(audio) - stop

    if start < end:
        trimmed = audio[start:end]
        trimmed.export(input, format="wav")


def fix_csv(file):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        rows = []
        last_init_time = None
        for row in reader:
            if row[5] == 'init':
                last_init_time = float(row[3])
            else:
                rows.append(row)

        if last_init_time is not None:
            for row in rows:
                row[3] = str(float(row[3]) - last_init_time)  # Replace with new time

    # Remove init rows
    with open(file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(rows)


def split_wav(input, stem, path, split_times, titles):
    audio = AudioSegment.from_wav(input)

    # horizontal: 22 seconds
    # forward/back -  5 seconds
    # turn- 2.5 seconds
    # rest- 1 second
    # descent- 2.5 seconds
    start_time = 0

    for i, end_time in enumerate(split_times):
        split_audio = audio[start_time:end_time]
        split_audio.export(f"{path}\\split\\{stem}_{titles[i]}.wav", format="wav")
        start_time = end_time

    # Last segment
    if start_time < len(audio):
        split_audio = audio[start_time:]
        split_audio.export(f"{path}\\split\\{stem}_descent.wav", format="wav")


if __name__ == '__main__':

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    path = project_root / f"recordings/noise_move"

    mydir = Path(path)
    for file in mydir.glob('*.wav'):
        # Trim start and end of wav-
        # trim_wav(file, 8000, 0) #first 8 seconds, last 0 seconds

        # Split wav into different actions- saved in sub-folder split
        # Horizontal flight:
        split_times = [3000, 4000, 9000, 10000, 12500, 13500, 18500, 19500]
        titles = ["ascent", "sleep1", "forward", "sleep2", "turn", "sleep3", "back", "sleep4"]
        # Idle:
        # split_times = [3000, 38000]
        # titles = ["ascent", "hover"]
        # split_wav(file, file.stem, path, split_times, titles)

        # Create spectogram	from .wav
        sample_rate, samples = wavfile.read(file)
        # sampling frequency = 44100 Hz

        # nperseg = 256 # window size only if window is not tuple/string. otherwise defaults to 256
        # noverlap = 256 # overlapping points, default- nperseg // 8
        # nfft = 256 # fft size (zero padding), default- nperseg

        # window: default- ('tukey',0.5) [second value is alpha- 0 for rectangular window, 1 for hann window]
        # instead of tuple, can write 'blackman', 'hamming' etc
        # second part is the window size, default- 256
        window = signal.get_window('hamming', 2048)

        # Apply high-pass filter to focus on fly frequencies
        samples = high_pass_filter(samples, sample_rate)

        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, window=window)

        plt.pcolormesh(times, frequencies, spectrogram)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        # Save spectogram plot as a png
        plt.savefig(path + file.stem + '_specto.png')
        plt.close()

    for file in mydir.glob('*.csv'):
        x = []
        y = []
        z = []
        t = []
        lab = []

        # Removes init tasks from csv, and adjusts sample times
        fix_csv(file)

        with open(file, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')

            for row in plots:
                x.append(float(row[0]))
                y.append(float(row[1]))
                z.append(float(row[2]))
                t.append(float(row[3]))
                lab.append(row[5])

        x_np = np.array(x)
        y_np = np.array(y)
        z_np = np.array(z)
        t_np = np.array(t)
        lab_np = np.array(lab)

        label_colors = {"sleep": 'red', "forward": 'blue', "turn": 'green', "back": 'yellow', "done": 'orange'}

        # 3-D projection
        ax = plt.axes(projection='3d')

        ax.plot3D(x, t, z, 'green')
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('t', labelpad=20)
        ax.set_zlabel('z', labelpad=20)

        # Save fly plot as a png
        plt.savefig(path + file.stem + '_path_x_z.png')
        plt.close()

        ax = plt.axes()

        line_seg_x = {}

        for label in np.unique(lab_np):
            indices = np.where(lab_np == label)
            line_seg_x[label] = {'t': t_np[indices], 'x': x_np[indices]}

        for label, segment in line_seg_x.items():
            plt.scatter(segment['t'], segment['x'], color=label_colors.get(label, 'black'), s=2, label=label)

        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.legend()

        # Save fly plot as a png
        plt.savefig(path + file.stem + '_path_x.png')
        plt.close()

        ax = plt.axes()

        ax.plot(t, y, 'green')
        ax.set_xlabel('t', labelpad=20)
        ax.set_ylabel('y', labelpad=20)

        # Save fly plot as a png
        plt.savefig(path + file.stem + '_path_y.png')
        plt.close()

        ax = plt.axes()

        ax.plot(t, z, 'green')
        ax.set_xlabel('t', labelpad=20)
        ax.set_ylabel('z', labelpad=20)

        # Save fly plot as a png
        plt.savefig(path + file.stem + '_path_z.png')
        plt.close()
