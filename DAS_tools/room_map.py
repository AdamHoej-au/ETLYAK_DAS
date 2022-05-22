import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn-colorblind")
plt.style.use("arh")


def view_placements(mic_array, source0, source1):
    speakers = [source0, source1]
    N = len(mic_array)
    fig, ax = plt.subplots(figsize=[10, 10], facecolor="w")

    # Plot lydkilder
    for i, speaker in enumerate(speakers):
        ax.plot(
            speaker[0],
            speaker[1],
            "o",
            label=f"Kilde {i+1}\n({speaker[0]}, {speaker[1]})",
        )
    # Plot mikrofoner
    ax.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
    for i,mic in enumerate(mic_array):
        ax.plot(
            mic[0],
            mic[1],
            "s",
            label=f"Mikro. {i+1}\n({mic[0]:01.2f}, {mic[1]})",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.legend(loc="upper center",ncol=2)
    ax.vlines(speakers[0][0], color="k", linestyle="--",ymin=0,ymax=speakers[0][1])
    # arrows between all speakers and all mikrofoner
    for speaker in speakers:
        ax.vlines(speaker[0], color="k", linestyle="--",ymin=0,ymax=speaker[1])
        for mic in mic_array:
            ax.arrow(
                speaker[0],
                speaker[1],
                mic[0] - speaker[0],
                mic[1] - speaker[1],
                head_width=0.00,
                head_length=0.0,
                linewidth=0.5,
            )
    ax.hlines(0, xmin=speakers[0][0],xmax=speakers[1][0], color="k", linestyle="--")

    ax.set_title(f"Placering af {N} mikrofoner og {len(speakers)} lydkilder")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
