import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import scipy.signal as sig
import glob
import soundfile as sf
import sounddevice as sd

plt.style.use("seaborn-colorblind")
plt.style.use("arh")


import matplotlib.ticker as ticker

formatEng = ticker.EngFormatter(unit="Hz")
formatdB = ticker.EngFormatter(unit="dB")


def distance(a, b):
    """Distancen mellem to punkter

    Args:
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(a) != 3 or len(b) != 3:
        a[2] = 0
        b[2] = 0
    return round(
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5, 3
    )


def time_delay(a, b, speed_of_sound=343):
    """
    Args:

    a (list): Mikrofon
    b (list): Kilde

    Returns:
        float: Tidsforskydning mellem de to punkter
    """

    if len(a) != 3 or len(b) != 3:
        a[2] = 0
        b[2] = 0
    time_delay = (distance(a, b) / speed_of_sound) * 1000
    return round(time_delay, 3)  # tidsforsinkelse i ms


def microphone_array(spacing, N, height=0):
    """Mikrofonarray"""
    # array of N elements from -x to x with spacing
    length = spacing * (N - 1)

    x = np.round(np.linspace(-length / 2, length / 2, N), 1)
    mic_array = []
    for idx, x_pos in enumerate(x):
        mic_array.append([x_pos, 0, height])
    return np.round(mic_array, 3)


def speaker(x, y, z=0):
    """Speaker"""
    return [x, y, z]


def DAS_calculator_3d_steer(mic_array, frequencies, angles, speed_of_sound=343.3):
    # omskrevet fra c til python - http://www.labbookpages.co.uk/audio/beamforming/delaySum.html
    num_mics = len(mic_array)

    N = len(frequencies)
    # distance = len(mic_array)
    output = np.zeros((N, N))
    realSum = np.zeros((N, N))
    imagSum = np.zeros((N, N))

    for mic_idx in range(num_mics):
        position = mic_array[mic_idx][0]
        # distance = np.sqrt(
        #     (source[0] - mic_array[mic_idx][0]) ** 2
        #     + (source[1] - mic_array[mic_idx][1]) ** 2
        #     + (source[2] - mic_array[mic_idx][2]) ** 2
        # )
        delay = position * np.sin(np.deg2rad(angles)) / speed_of_sound
        # delay = distance / speed_of_sound

        realSum += np.cos(2 * np.pi * frequencies * delay)
        imagSum -= np.sin(2 * np.pi * frequencies * delay)
    output = np.sqrt(realSum ** 2 + imagSum ** 2) / num_mics

    return output


def DAS_calculator(mic_array, freq, angle_res=500, speed_of_sound=343.3):
    # omskrevet fra c til python - http://www.labbookpages.co.uk/audio/beamforming/delaySum.html
    num_mics = len(mic_array)
    output = np.zeros(angle_res)

    for idx in range(angle_res):
        angle = -90 + 180 * idx / (angle_res - 1)
        angle_rad = angle * np.pi / 180

        realSum = np.zeros(angle_res)
        imagSum = np.zeros(angle_res)

        for mic_idx in range(num_mics):
            position = mic_array[mic_idx][0]
            delay = position * np.sin(angle_rad) / speed_of_sound

            realSum[idx] += np.cos(2 * np.pi * freq * delay)
            imagSum[idx] += np.sin(2 * np.pi * freq * delay)

        output[idx] = np.sqrt(realSum[idx] ** 2 + imagSum[idx] ** 2) / np.sqrt(num_mics)

    return output


def DAS_plot(das_output, angle_res=500, freq=1000, num_mics=3):
    theta = np.linspace(-90, 90, angle_res)

    theta *= np.pi / 180
    ax = plt.figure().subplots(subplot_kw={"projection": "polar"})
    ax.plot(theta, 20 * np.log10(das_output))
    ax.set_rlim(-60, 0)
    ax.set_title(f"DAS-pattern for {num_mics} microphones at {freq} Hz")
    ax.set_xlabel("Angle [deg]")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)


def DAS_polar(spacing, N, frequencies=[125, 250, 500, 1000, 2000, 4000]):
    """_summary_

    Args:
        mic_array (_type_): _description_
        frequencies (list, optional): _description_. Defaults to [125, 250, 500, 1000, 2000, 4000].
    """
    mic_array = microphone_array(spacing, N)
    angle_resolution = 180
    theta = np.linspace(-90, 90, angle_resolution)
    theta *= np.pi / 180

    das_output = np.zeros((len(frequencies), angle_resolution))

    fig, ax = plt.subplots(
        1,
        len(frequencies),
        subplot_kw={"projection": "polar"},
        figsize=(20, 5),
        facecolor="w",
    )
    for idx, frequency in enumerate(frequencies):
        das_output[idx] = DAS_calculator(
            mic_array, freq=frequency, angle_res=angle_resolution
        )
        ax[idx].plot(
            theta, 20 * np.log10(das_output[idx]), label=f"{frequency} Hz", lw=2
        )
        ax[idx].set_title(f"{frequency} Hz")
    # fig.suptitle(f"Frekvensrespons for {N} mikrofoner, med {spacing*100}cm afstand")
    for ax_ in ax:
        ax_.set_xlabel("Vinkel [deg]")
        # ax_.set_xlim(np.deg2rad(-90),np.deg2rad(90))
        ax_.set_theta_zero_location("N")
        ax_.set_theta_direction(-1)
        ax_.set_rlim(-60, 0)
    fig.tight_layout()
    fig.savefig(f"DAS_polar_{N}_{spacing*100}.eps")
    return mic_array


def _angle_calculation(source, mic_array):
    angles = np.zeros((len(mic_array), 1))
    side = "left"
    for idx, mic in enumerate(mic_array):
        dx = abs(mic[0] - source[0])
        direct_path = distance(source, mic)
        angles[idx] = np.arccos(dx / direct_path) * 180 / np.pi
        # print(f"dx: {dx}")
        # print(f"direct: {direct_path}")
        # print(f"Arrival angle: {angles[idx]}")
    if angles[0] < angles[1]:
        side = "right"
    print(f"The source is placed to the {side} of the array.")
    return angles, side


def delay_between_microphones(source, mic_array, fs=48e3):
    """Tidsforskydning mellem mikrofoner"""
    angles, side = _angle_calculation(source, mic_array)
    spacing = abs(mic_array[0][0] - mic_array[1][0])
    n = len(mic_array)
    temp = np.linspace(0, spacing * (n - 1), n)
    weight = np.linspace(0.8, 1, n)
    mic_array_x = mic_array.copy()
    # replace the [idx][0] with the correct value
    angle = angles[0]
    for idx, mic in enumerate(mic_array):
        mic_array_x[idx][0] = temp[idx]
    if side == "left":
        mic_array_x = mic_array_x[::-1]
        weight = weight[::-1]
    # if side == "right":
    # angles = -angles
    shift_samples = np.zeros((len(mic_array_x)))
    ts = 1 / fs
    for idx, (mic) in enumerate(mic_array_x):
        shift_samples[idx] = ((((mic[0]) * np.cos(angle))) / 343.3) / ts
        print(
            f"Mic:\t\t{mic}\n"
            f"angle: {angle}\n"
            f"extra distance:{((mic[0])*np.cos(np.deg2rad(angle)))}m\n"
            # f"extra time:{((((mic[0])*np.cos(np.deg2rad(angle))))/343.3)*1e3}ms\n"
            # f"sample_diff:\t{((((mic[0])*np.cos(np.deg2rad(angle))))/343.3)/ts}samples\n"
        )
    return shift_samples, weight


def FIR_calculate(source, mic_array, fs=48e3, FIR_taps=25):
    """Tidsforskydning af lyd"""
    shift_samples, weight = delay_between_microphones(source, mic_array, fs=fs)
    int_delay = np.floor(shift_samples)
    fract_delay = shift_samples % 1
    center_tap = FIR_taps // 2
    print(f"Int delay: {int_delay}")
    print(f"Fract delay: {fract_delay}")
    coeffs = np.zeros((len(int_delay), (FIR_taps)))
    for i, delay in enumerate(fract_delay):
        if delay == 0:
            continue
        for t in range(FIR_taps):
            x = t - delay
            sinc = np.sin(np.pi * (x - center_tap)) / (np.pi * (x - center_tap))
            coeffs[i, t] = sinc
            coeffs[i, :] *= np.blackman(FIR_taps)
    # fig, ax = plt.subplots(2,figsize=[10, 10], facecolor="w")
    # freq = np.linspace(0,fs/2,coeffs.shape[1])
    # for idx,coeff in enumerate(coeffs):
    #     coeff_fft = np.fft.fft(coeff)
    #     coeff_fft = 20*np.log10(abs(coeff_fft))
    #     ax[0].plot(coeff,label=f"Mikro. {idx+1}",lw=2,marker="s",markersize=8)
    #     ax[1].semilogx(freq,coeff_fft,label=f"Mikro. {idx+1}",lw=2)
    #     ax[1].xaxis.set_major_formatter(formatEng)

    # ax[0].set_xlabel("Sample [n]")
    # ax[0].set_ylabel("Amplitude")
    # ax[0].set_title("FIR filter koefficienter")
    # ax[1].set_xlabel("Frekvens [Hz]")
    # ax[1].set_ylabel("Amplitude [dB]")
    # ax[1].set_xlim(20,fs//2)
    # ax[1].grid(which="both")
    # # vertical grid
    # ax[0].xaxis.grid(True, linestyle='-', which='major', color='grey')
    # # ax[0].grid()
    # ax[0].legend()
    return int_delay, coeffs, weight, shift_samples


def sound_shift(mic_inputs, mic_array, source_position):
    """Tidsforskydning af lyd"""
    int_delay, coeffs, weight, shift_samples = FIR_calculate(source_position, mic_array)
    shifted_inputs = mic_inputs.copy()
    # for mic in mic_inputs:
    for idx, (delay, mic, coeff) in enumerate(
        # zip(int_delay[::-1], shifted_inputs, coeffs[::-1])
        zip(int_delay, shifted_inputs, coeffs)
    ):
        shifted_inputs[idx] = np.roll(mic, int(delay))
        if delay := 0:
            shifted_inputs[idx] = np.roll(
                shifted_inputs[idx], -((coeffs.shape[1] - 1) // 2)
            )
            shifted_inputs[idx] = np.convolve(shifted_inputs[idx], coeff, mode="same")
            # Correction of FIR
        # shifted_inputs[idx] = shifted_inputs[idx]
        shifted_inputs[idx] *= weight[idx]
    return shifted_inputs, shift_samples[::-1]


def sound_sum(sound_input, cutoff=0):
    """Sum af lyd"""
    if cutoff != 0:
        b, a = sig.butter(4, cutoff, "high", fs=48e3)
    sum = np.zeros((len(sound_input[0])))

    for mic in sound_input:
        if cutoff == 0:
            sum += mic
        else:
            sum += sig.filtfilt(b, a, mic)
    # sum /= sound_input.shape[0]  # normalize to the number of mics
    sum /= np.sqrt(sound_input.shape[0])  # normalize to the number of mics
    return sum


def source_viewer(src, fs, title="", filename=""):

    if len(src.shape) > 1:
        src = src[:, 0]

    # src = src / np.max(np.abs(src))
    N = len(src)
    t = np.arange(N) / fs
    src_fft = rfft(src)
    # window
    window = np.blackman(N)
    freq = rfftfreq(N, 1 / fs)
    src_mag = np.abs(src_fft) /  N

    fig = plt.figure(figsize=(14, 10), facecolor="w")
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax1.plot(t, src)
    ax1.set_xlabel("Tid [s]")
    ax1.set_ylabel("Amplitude")
    ax1.grid()
    ax2.specgram(src, NFFT=4096, Fs=fs, noverlap=2048, scale="dB")
    ax2.set_yscale("log")
    ax2.set_ylim(200, 20e3)
    # ax2.set_xlim(0, 2)
    ax2.set_xlabel("Tid [s]")
    ax2.yaxis.set_major_formatter(formatEng)
    ax3.semilogx(freq, src_mag)
    ax3.set_ylabel("Magnitude")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_xlim(100, fs / 2)
    ax3.grid()
    ax3.xaxis.set_major_formatter(formatEng)
    ax1.set_title("Lydkilde")
    ax2.set_title("Spektogram")
    ax3.set_title("Magnitude")
    fig.suptitle(title)
    if filename != "":
        fig.savefig(f"figs/{filename}.eps", format="eps", transparent=False)


def shift_viewer(
    mic_inputs,
    mic_array,
    speaker_location,
    start=1.418,
    stop=1.43,
    axv=1.418,
    fs=48e3,
    filename="",
):
    shifted_spk, shift_samples = sound_shift(mic_inputs, mic_array, speaker_location)
    fig, ax = plt.subplots(
        1, 2, figsize=(14, 8), sharex=True, sharey=True, facecolor="w"
    )
    N = len(mic_inputs[0])
    plt_space = np.max(np.abs(mic_inputs)) * 1.5
    num_mics = len(mic_array)
    t = np.linspace(0, N / fs, N)
    ticks = np.arange(0, plt_space * num_mics, plt_space)
    color = ["b", "g", "r", "c", "m", "y", "k"]
    for idx, (mic, org) in enumerate(zip(shifted_spk, mic_inputs)):
        ax[1].plot(
            t,
            mic + plt_space * idx,
            color=f"{color[idx]}",
            lw=4,
            label=f"Forskudt: {shift_samples[idx]:.2f} samples",
        )
        ax[0].plot(
            t,
            org + plt_space * idx,
            color=f"{color[idx]}",
            lw=4,
            # label=f"Original : {idx}",
        )
        ax[0].set_yticks(ticks)
        ax[1].set_yticks(ticks)
    ax[1].set_title("Tidsforskudt signaler")
    ax[0].set_title("Original signaler")
    ylabels = ["Mic1", "Mic2", "Mic3", "Mic4", "Mic5"]
    ax[0].set_yticklabels(ylabels)
    ax[1].set_yticklabels(ylabels)
    ylabels = ["Mic %i" % i for i in range(1, 5)]
    for axs in ax:
        axs.axvline(x=axv, color="k", ls="--")
        axs.set_xlabel("Tid [s]")
        axs.set_xlim(start, stop)
        axs.grid(which="both")
    fig.suptitle("Tidsforskydning af signaler")
    fig.tight_layout()

    if filename != "":
        fig.savefig(f"figs/{filename}.eps")

    return shifted_spk


def load_inputs(path="", wildcard=""):
    data, fs = sf.read(f"{path}/Mic1{wildcard}.wav")
    files = glob.glob(f"{path}/Mic*{wildcard}.wav")
    mic_inputs = np.zeros((len(files), len(data)))
    for idx, name in enumerate(files):
        data = sf.read(name)[0]
        if data.shape != mic_inputs[0].shape:
            mic_inputs[idx] = data[:, 0]
            continue
        mic_inputs[idx] = data
    return mic_inputs, fs
