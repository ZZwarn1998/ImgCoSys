import os
import cv2
import csv
import numpy as np
from typing import List
from imageEncoder import ImageEncoder
from imageDecoder import ImageDecoder
from skimage.metrics import peak_signal_noise_ratio


def psnr(raw: np.ndarray,
         noise: np.ndarray
         ) -> float:
    """Return peak signal-to-ratio of `raw` and `noise`.

    Args:
        raw: Raw picture.
        noise: Depressed picture.

    Returns:
        Peak signal-to-noise ratio between `raw` and `noise`.
    """
    psnr = peak_signal_noise_ratio(raw, noise)
    return psnr


def start(names: List[str],
          T: int = 5,
          steps: int = 51
          ) -> None:
    """Start compression and decompression process.

    Args:
        names: The list of image's names.
        T: The number of DWT or IDWT.
        steps: Step size of quantization.
    Raises:
        ValueError1: `names` is empty.
        ValueError2: `T` is lower than or equal to 0.
        ValueError3: `steps` is lower than or equal to 0.
    """
    if len(names) == 0:
        raise ValueError("`names` is empty.")
    if T <= 0:
        raise ValueError("`T` is lower than or equal to 0.")
    if steps <= 0:
        raise ValueError("`steps` is lower than or equal to 0.")

    # Carry out experiment
    for i in range(0, len(names)):
        name = names[i]  # Image name
        path = ".\\pic\\" + name  # The path of original image
        data_path = ".\\data\\csv\\" + name.split('.')[0] + ".csv"  # The storage path of csv file
        bin_path = ".\\out\\image.bit"  # The path of binary file
        prefix = ".\\data\\pic\\" + name.split('.')[0]  # The prefix of storage path of decompressed image

        print("{:-^33}".format(name))
        # print(len(">q=41 T=5 D=25.65 R=0.24 CR=33.04"))
        # Initialization of csv writer
        file = open(data_path, "w", newline='')
        wt = csv.writer(file)
        
        # Write header row into the csv file
        wt.writerow(["q", "D", "R", "CR"])

        # Run
        for q in reversed(range(40, steps)):
            raw_size = os.path.getsize(path)  # The size of raw file
            storage_path = prefix + "_" + str(q) + ".png"  # Storage path

            # Compress image
            encoder = ImageEncoder(path, q, T)
            encoder.run()

            # Decompress binary file
            decoder = ImageDecoder(bin_path, q, name)
            img = decoder.run()

            # Obtain necessary indicators
            compressed_size = os.path.getsize(".\\out\\image.bit")  # The size of compressed binary file
            PSNR = psnr(encoder.img, img)  # Peak signal-to-ratio
            R = compressed_size * 8 / (decoder.H * decoder.W)  # Bitrate
            CR = raw_size / compressed_size  # Compression Ratio
            row = [q, PSNR, R, CR]

            # Write data
            wt.writerow(row)
            cv2.imwrite(storage_path, img)

            # Print indicators
            print(">q=%d" % q, "T=%d" % T, "D=%.2f" % PSNR, "R=%.2f" % R, "CR=%.2f" % CR)

            # Show decompressed img
            # cv2.imshow(f"q_{q}", img)
            # cv2.waitKey()

        # Disconnect the file
        file.close()


if __name__ == "__main__":
    names = ["image1.512", "image2.512", "image3.512", "image4.512", "image5.512"]  # The list of image's names
    T = 5  # The number of DWT or IDWT
    steps = 51  # Quantization step size.

    # Carry out experiment
    start(names, T, steps)
