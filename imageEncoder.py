import copy
import math
import os
import sys
import numpy as np
from waveletTransformer import WaveletTransformer
from classesAbtZeroTree import ZeroTreeEncoder
from collections import Counter
from bitarray import bitarray
import huffman
from typing import List, Tuple, Mapping

# Start of the Image
SOI = bitarray()
SOI.frombytes(b'\xff\xd8')
# Start of Scan 1
SOS1 = bitarray()
SOS1.frombytes(b'\xff\xd9')
# Start of Scan 2
SOS2 = bitarray()
SOS2.frombytes(b'\xff\xda')
# End of the Image
EOI = bitarray()
EOI.frombytes(b'\xff\xdb')


class ImageEncoder:
    """A class is used for transforming an image into binary file.

    Attributes:
        img (np.ndarray): Raw image.
        q (int): An integer represents quantization step size.
        T (int): An integer represents number of DWT or IDWT.
        dwt_img (np.ndarray): Image undergoing DWT.
        q_img (np.ndarray): Image undergoing Quantization.

    """
    def __init__(self,
                 img_path: str,
                 q: int,
                 t: int
                 ) -> None:
        """Constructor of class `ImageEncoder`.

        Args:
            img_path: A string represents the path of original image.
            q: An integer represents quantization step size.
            t: An integer represents number of DWT or IDWT.
        Raises:
            ValueError: `img_path` is not available.
        """
        if not os.access(img_path, os.F_OK):
            raise ValueError("`img_path` is not available.")
        self.img = self.readRawFile(img_path)
        self.q = q
        self.T = t
        self.dwt_img = np.zeros((0, 0))
        self.q_img = np.zeros((0, 0))

    def run(self) -> None:
        """Compress image and generate binary file"""
        # DWT
        dwt_img = self.DWT()
        self.dwt_img = dwt_img

        # Quantization
        q_img = self.quantization(dwt_img)
        self.q_img = copy.copy(q_img)

        # Get the lowest frequency sub-band
        (h, w) = (self.img.shape[0] // 2 ** self.T, self.img.shape[1] // 2 ** self.T)
        LL = copy.copy(q_img[0:h, 0:w])

        # Raster scan
        LL_seq = LL.reshape(-1)

        # Prediction
        LL_seq = self.prediction(LL_seq)

        # Run-length coding
        LL_sym_seq, LL_size_freq = self.run_length_coding(LL_seq)

        # Encode symbol sequence of the lowest frequency sub-band to 01 string and count all types of items
        LL_bits_str = self.ll_symseq2bitsstr(LL_sym_seq, LL_size_freq)

        # Zero-tree scan
        coeffs = self.split_quantization_area()
        zerotree_encoder = ZeroTreeEncoder(coeffs)
        H_sym_seq, H_sym_freq, nonzero_lis = zerotree_encoder.travel()
        nonzero_lis_length = len(nonzero_lis)
        nonzero_bits_str, nonzero_size_freq = self.nozerolis2bitsstr(nonzero_lis)
        H_bits_str = self.h_symseq2bitsstr(H_sym_seq, H_sym_freq)

        # Output bits string into binary file
        output = ".\\out\\image.bit"
        file = open(output, "wb")

        BINFILE = bitarray()

        # Start of the image
        BINFILE.extend(SOI)

        # Basic attributes about image
        (H, W) = self.img.shape
        bH = '{:032b}'.format(H)
        bW = '{:032b}'.format(W)
        bQ = '{:032b}'.format(self.q)
        bT = '{:032b}'.format(self.T)
        str1 = bH + bW + bQ + bT
        bstr1 = bitarray(str1)
        BINFILE.extend(bstr1)

        # Start of the scan 1
        BINFILE.extend(SOS1)

        # Binary information about the lowest frequency sub-band
        LL_size_freq_length_str = "{:032b}".format(len(LL_size_freq))
        LL_size_freq_str = self.sym_freq2str(LL_size_freq)
        str2 = LL_size_freq_length_str + LL_size_freq_str + LL_bits_str
        bstr2 = bitarray(str2)
        BINFILE.extend(bstr2)

        # Start of the scan 2
        BINFILE.extend(SOS2)

        # Binary information about other sub-bands
        nonzero_lis_length_str = "{:032b}".format(nonzero_lis_length)
        nonzero_size_freq_length_str = "{:032b}".format(len(nonzero_size_freq))
        nonzero_size_freq_str = self.sym_freq2str(nonzero_size_freq)
        str3 = nonzero_lis_length_str + nonzero_size_freq_length_str + nonzero_size_freq_str + nonzero_bits_str
        bstr3 = bitarray(str3)
        BINFILE.extend(bstr3)
        H_sym_freq_length_str = "{:032b}".format(len(H_sym_freq))
        H_sym_freq_str = self.sym_freq2str(H_sym_freq)
        H_sym_seq_length_str = "{:032b}".format(len(H_sym_seq))
        str4 = H_sym_seq_length_str + H_sym_freq_length_str + H_sym_freq_str + H_bits_str
        bstr4 = bitarray(str4)
        BINFILE.extend(bstr4)

        # End of the image
        BINFILE.extend(EOI)
        BINFILE.tofile(file)

        # Disconnect the file
        file.close()

        return

    def sym_freq2str(self, sym_freq: dict) -> str:
        """Transform symbol-frequency pairs into a string which is made up of 0 and 1.

        Args:
            sym_freq (dict):

        Returns:
            bstr (str): A string made of character `0` and `1`.
        """
        bstr = ""
        for item in sym_freq:
            key = item[0] if type(item[0]) != type('a') else ord(item[0])
            val = item[1]
            bstr = bstr + "{:08b}".format(key) + "{:032b}".format(val)

        return bstr

    def split_quantization_area(self) -> List[List[np.ndarray]]:
        """Split image undergoing DWT, Prediction and Quantization into coefficient matrices.

        Returns:
            A list containing coefficient matrices corresponding to sub-band areas.

        """
        func = lambda x:  x // 2 ** levels
        levels = self.T
        q_img = copy.copy(self.q_img)  # Image undergoing quantization
        (H, W) = self.img.shape  # Height and Width of raw image
        w = func(W)
        h = func(H)

        coeffs = []
        for i in range(levels + 1):
            if i == 0:
                coeffs.append(q_img[: h, : w])
            else:
                coeffs.append([q_img[:h, w: 2 * w],
                               q_img[h:2 * h, w: 2 * w],
                               q_img[h: 2 * h, : w]])
                h *= 2
                w *= 2

        return coeffs

    def readRawFile(self, path: str) -> np.ndarray:
        """Read raw image.

        Args:
            path: The path of selected image.

        Returns:
            The image.

        """
        img = None
        try:
            img = np.fromfile(path, dtype="uint8")
            img = img.reshape(512, 512)
        except Exception as e:
            print(e)
        return img

    def DWT(self) -> np.ndarray:
        """Discrete Wavelet Transformation.

        Returns:
            Image undergoing DWT.

        """
        wtf = WaveletTransformer(self.img, self.T, self.img.shape, mode="a")
        img = wtf.DWT()
        return img

    def quantization(self, dwt_img: np.ndarray) -> np.ndarray:
        """Quantization.

        Args:
            dwt_img: Image undergoing DWT.

        Returns:
            Image undergoing Quantization.

        """
        q_img = np.round(dwt_img / self.q)
        return q_img

    def prediction(self, seq: np.ndarray) -> np.ndarray:
        """Add prediction to the lowest frequency sub-band. Find the corresponding backward difference array of input
        one-dimensional array.

        Args:
            seq: An one-dimensional array generated by roster scan.

        Returns:
            An one-dimensional array corresponding to backward difference array of `seq`.

        """
        for i in reversed(range(1, len(seq))):
            seq[i] = seq[i] - seq[i - 1]
        return seq

    def val2size(self, val: int) -> int:
        """Obtain the exact bit length of binary form of integer `val`.

        Args:
            val (int): An integer.

        Returns:
            size (int): The exact bit length of binary form of integer `val`.

        """
        size = math.ceil(math.log2(abs(val) + 0.1))
        return size

    def amp2bitstr(self,
                   size: int,
                   amp: int) -> str:
        """Use size and amplitude to obtain bit string.

        Args:
            size (int): An integer representing `size` in Run-length Coding.
            amp (int): An integer representing `amplitude` in Run-length Coding.

        Returns:
            bitstr (str): A string made by `0` and `1`.

        """
        try:
            if amp == 0:
                raise Exception
        except Exception as e:
            sys.exit("ZERO doesn't have amplitude. Please check your code.")

        index = -1
        if amp > 0:
            index = amp
        elif amp < 0:
            index = amp + 2**size - 1
        fstr = '{:0' + str(size) + 'b}'
        bitstr = fstr.format(index)

        return bitstr

    def run_length_coding(self, seq: np.ndarray) -> Tuple[List[Tuple[int, ...]], List[Mapping[int, int]]]:
        """Run-length coding.

        There are three types of tuples. They are (`run-length`, `size`, `amplitude`), (0, 0) and (255,).
        `run-length` stands for the number of zeros between two nonzero numbers. Assuming that we have a nonzero number,
        `num`, `size` is calculated by the function `self.val2size(num)` and the value of `amplitude` is equal to `num`.
        (0, 0) means the end of sequence of run-length coding. (255, 0) means that there has been 255 zeros and we need
        to create a new tuple to restart our coding. (255, 0) is created to fix the bit length of `size` at 8 bits.

        Args:
            seq: An one-dimensional array undergoing prediction.

        Returns:
            A tuple (sym_seq, size_freq), where sym_seq is a list used for collecting all tuples of run-length representations, and size_freq is a list contains all mappings from `size`s to corresponding frequencies.

        """
        # Start run-length coding
        sym_seq = []  # A list used for collecting all tuples of run-length representations
        collect = []  # A list used for collecting `size`
        cursor = 0  # A integer standing for current position of scanning.
        hd = int(seq[cursor])  # Header
        DC = (0, self.val2size(hd), hd)  # The tuple of header
        collect.append(self.val2size(hd))   # Add the size of `hd` into list `collect`
        nozero_loc = np.nonzero(seq)[0]  # Find the index of nearest nonzero number from the front of the list `seq` to
        # the back.
        sym_seq.append(DC)  # Add tuple `DC` into list `sym_seq`

        # Continue run-length coding
        for i, cursor in enumerate(nozero_loc[:-1]):
            next_cursor = nozero_loc[i + 1]
            zero_num = next_cursor - cursor - 1

            # If the number of zeros is less than 255
            if zero_num < 255:
                val = np.int(seq[next_cursor])
                size = self.val2size(val)
                collect.append(size)
                mark = (zero_num, size, val)
                sym_seq.append(mark)
            else:  # If not
                k = zero_num // 255
                r = zero_num % 255
                val = np.int(seq[next_cursor])
                size = self.val2size(val)
                collect.append(size)
                mark = []
                for i in range(k):  # The number of zero is more than or equal to 255
                    mark.append((255,))
                mark.append((r, size, val))
                sym_seq.extend(mark)

        # End of run-length coding
        if nozero_loc[-1] < len(seq) - 1:
            sym_seq.append((0, 0))
            collect.append(0)

        # Count the frequency of each `size`s
        counter = Counter(collect)
        # Sort all items of `counter` from high frequency to low frequency
        sizes = sorted(counter.keys(), key=lambda x: x)
        freq = []
        for item in sizes:
            freq.append(counter.get(item))
        size_freq = list(zip(sizes, freq))

        return sym_seq, size_freq

    def ll_symseq2bitsstr(self,
                          seq: List[Tuple[int, ...]],
                          sym_freq: List[Mapping[int, int]]
                          ) -> str:
        """Change symbol-frequency pairs in the lowest frequency sub-band into a string which is made up of 0 and 1.

        Args:
            seq: A list containing tuples of Run-length Coding.
            sym_freq: A list contains all mappings from `size`s to corresponding frequencies.

        Returns:
            A string made by character `0` and `1`.

        """
        cdbk = huffman.codebook(sym_freq)  # Codebook
        bits_str = ""  # Bit string

        # Change each tuple into bit string.
        for item in seq:
            bin_str = ""
            # If (run length, size, amplitude)
            if len(item) == 3:
                run_length = item[0]
                size = item[1]
                amplitude = item[2]

                binstr_rl = '{:08b}'.format(run_length)
                binstr_s = cdbk.get(size)
                binstr_a = self.amp2bitstr(size, amplitude)

                bin_str = binstr_rl + binstr_s + binstr_a
            elif len(item) == 2:  # If (0, 0)
                run_length = item[0]
                size = item[1]

                binstr_rl = '{:08b}'.format(run_length)
                binstr_s = cdbk.get(size)

                bin_str = binstr_rl + binstr_s
            elif len(item) == 1:  # If (255,)
                run_length = item[0]
                binstr_rl = '{:08b}'.format(run_length)

                bin_str = binstr_rl
            bits_str = bits_str + bin_str
            pass
        return bits_str

    def h_symseq2bitsstr(self,
                         seq: List[str],
                         sym_freq: List[Mapping[str, int]]
                         ) -> str:
        """Change mark-the frequency of the mark pairs in other frequency sub-band into a string which is made up of 0
        and 1.

        Args:
            seq: A list containing all marks.
            sym_freq:A list contains all mappings from marks to the corresponding frequencies.

        Returns:
            A string made by character `0` and `1`.

        """
        cdbk = huffman.codebook(sym_freq)  # Codebook
        ba_val = []
        for val in cdbk.values():
            ba_val.append(bitarray(val))

        dic = dict(zip(cdbk.keys(), ba_val))
        ba = bitarray()
        ba.encode(dic, seq)
        bits_str = ''.join(ba.decode({'1': bitarray('1'), '0': bitarray('0')}))

        return bits_str

    def nozerolis2bitsstr(self, lis: List[int]) -> Tuple[str, List[Mapping[int, int]]]:
        """Transform list of nonzero numbers into a string which is made up of 0 and 1 and obtain symbol-frequency
        pairs.

        Args:
            lis: A list of nonzero integers.

        Returns:
            A tuple (bits_str, sym_freq), where bits_str is a string made by character `0` and `1`, and sym_freq is a list containing all mappings from nonzero numbers to corresponding frequencies.

        """
        sizes = []
        for val in lis:
            sizes.append(self.val2size(val))
        counter = Counter(sizes)

        syms = sorted(counter.keys(), key=lambda x: x)
        freq = []
        for sym in syms:
            freq.append(counter.get(sym))
        sym_freq = list(zip(syms, freq))
        cdbk = huffman.codebook(sym_freq)

        bits_str = ""
        for val in lis:
            size = self.val2size(val)
            binstr_s = cdbk.get(size)
            binstr_a = self.amp2bitstr(size, int(val))
            bits_str = bits_str + binstr_s + binstr_a

        return bits_str, sym_freq