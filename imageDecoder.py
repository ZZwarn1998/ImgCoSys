import cv2
import huffman
import numpy as np
from tqdm import tqdm
from bitarray import bitarray
from collections import deque
from typing import Tuple, List
from classesAbtZeroTree import ZeroTreeDecoder
from waveletTransformer import WaveletTransformer

# Start of the Image
SOI = bytes.fromhex("FFD8")
# Start of Scan 1
SOS1 = bytes.fromhex("FFD9")
# Start of Scan 2
SOS2 = bytes.fromhex("FFDA")
# End of the Image
EOI = bytes.fromhex("FFDB")


class ImageDecoder:
    """A class is used for decoding binary file and reconstructing image.

    Attributes:
        path (str): The path of binary file.
        q (int): Step size of Quantization.
        name (str): The name of image.
        H (int): The height of image.
        W (int): The width of image.
        h (int): The height of the lowest sub-band.
        w (int): The width of the lowest sub-band.
        T (int): An integer represents number of DWT or IDWT.
        img (np.ndarray): Raw image.
        dwt_img (np.ndarray): Image undergoing DWT.
        q_img (np.ndarray): Image undergoing Quantization.

    """
    def __init__(self,
                 file_path: str,
                 q: int,
                 img_name: str) -> None:
        """Constructor of class `ImageDecoder`.

        Args:
            file_path: A string represents the path of binary file.
            q: Quantization step.
            img_name: The name of image.

        """
        self.path = file_path
        self.q = q
        self.name = img_name
        self.H = None
        self.W = None
        self.h = None
        self.w = None
        self.T = None
        self.img = None
        self.dwt_img = None
        self.q_img = None

    def run(self) -> np.ndarray:
        """A method is used to run decoding process.

        Returns:
            An image.

        """
        # Read binary file
        coeffs, codes, non_zeros = self.readBinaryFile()

        # Rebuild zero tree and initialize coefficients
        decoder = ZeroTreeDecoder(coeffs)

        # Fill coefficients with zero tree
        decoder.refill(codes, non_zeros)

        # Obtain quantization image by synthesis matrices of coefficients
        q_img = self.synthesis_split_quantization_area(decoder.coeffs)

        # Inverse quantization
        self.dwt_img = self.i_quantization(q_img)

        # Inverse discrete wavelet transformation
        self.img = self.iDWT()

        return self.img

    def synthesis_split_quantization_area(self, coeffs: List[List[np.ndarray]]) -> np.ndarray:
        """Synthesis all sub-bands.

        Args:
            coeffs: A list containing coefficient matrices corresponding to sub-band areas.

        Returns:
            Image undergoing quantization.
        """
        q_img = np.zeros((self.H, self.W), np.float32)
        h = self.h
        w = self.w
        levels = self.T
        for i in range(levels + 1):
            if i == 0:
                q_img[0:h, 0:w] = coeffs[i]
            else:
                (q_img[0:h, w:2*w],
                 q_img[h:2*h, w:2*w],
                 q_img[h:2*h, 0:w]) = coeffs[i]
                h *= 2
                w *= 2
        return q_img

    def readBinaryFile(self) -> Tuple[List[List[np.ndarray]], List[str], List[int]]:
        """Read binary file and return necessary parameters used for reconstructing image.

        Returns:
            A tuple (coeffs, codes, non-zeros), where coeffs is a list containing coefficient matrices corresponding to sub-band areas, codes is a list containing all marks, and non_zeros is a list containing nonzero numbers.

        Raises:
            ValueError1: SOI marker mismatched.
            ValueError2: EOI marker mismatched.
        """
        cnt = 0  # An integer used to count bits
        LL_cdbk = None  # Codebook used to reconstruct LL area
        Non_cdbk = None  # Codebook used to reconstruct non-zero list
        Zt_cdbk = None  # Codebook used to reconstruct zero tree

        with open(self.path, "rb") as rf:
            soi = rf.read(2)  # Start of Image
            cnt += 2

            # Check the marker, SOI
            if soi != SOI:
                raise Exception("There isn't a SOI symbol. Please check your codes.")

            # obtain height(H), width(W), quantization step(q) and times of up sampling(T)
            H = int.from_bytes(rf.read(4), 'big')
            self.H = H
            W = int.from_bytes(rf.read(4), 'big')
            self.W = W
            q = int.from_bytes(rf.read(4), 'big')
            self.q = q
            T = int.from_bytes(rf.read(4), 'big')
            self.T = T

            # Obtain the shape of LL area
            self.h = H // 2 ** self.T
            self.w = W // 2 ** self.T

            cnt += 4 * 4
            sos1 = rf.read(2)  # Start of scan 1
            cnt += 2

            # Check the marker, SOS1
            if sos1 != SOS1:
                raise Exception("There isn't a SOI1 symbol. Please check your codes.")

            # Reconstruct LL_cdbk
            LL_size_freq_length = int.from_bytes(rf.read(4), 'big')
            cnt += 4
            LL_size_freq = []
            for i in range(LL_size_freq_length):
                size = int.from_bytes(rf.read(1), 'big')
                freq = int.from_bytes(rf.read(4), 'big')
                LL_size_freq.append((size, freq))
                cnt += 5

            LL_cdbk = huffman.codebook(LL_size_freq)

        rf.close()

        rf = open(self.path, "rb")
        ba = bitarray()
        ba.fromfile(rf)
        del ba[:cnt * 8]

        # Reverse key-value pairs of LL codebook
        rev_LL_cdbk = {value: key for (key, value) in LL_cdbk.items()}

        NEXT_SYMBOL = bitarray()
        NEXT_SYMBOL.frombytes(SOS2)

        LL_seq = []
        while (ba[:16] != NEXT_SYMBOL):
            # Obtain `run length`
            run_length = int("".join(ba[:8].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            del ba[:8]
            if run_length == 255:  # Encounter (255,)
                LL_seq.extend([0 for i in range(255)])
                continue

            # Obtain `size`
            cursor = 1
            prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            while (prefix not in rev_LL_cdbk.keys()):
                cursor += 1
                prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            size = rev_LL_cdbk.get(prefix)
            del ba[:cursor]

            if run_length == 0 and size == 0:  # Encounter (0, 0)
                zeros = [0 for i in range(self.h * self.w - len(LL_seq))]
                LL_seq.extend(zeros)
            else:  # Encounter (`run length`, `size`, `amplitude`)
                index = int("".join(ba[: size].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
                amplitude = self.size_index2amp(size, index)
                zeros = [0 for i in range(run_length)]
                LL_seq.extend(zeros)
                LL_seq.append(amplitude)
                del ba[:size]

        del ba[:16]

        LL_seq = np.array(LL_seq, dtype=np.float32)
        LL_q_img = self.i_prediction(LL_seq).reshape(self.h, self.w)

        coeffs = self.get_init_coeffs()
        coeffs[0] = LL_q_img

        NEXT_SYMBOL = bitarray()
        NEXT_SYMBOL.frombytes(EOI)

        non_zero_length = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
        del ba[: 32]
        non_zero_size_freq_length = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
        del ba[: 32]

        sizes = []
        freqs =[]
        for i in range(non_zero_size_freq_length):
            size = int("".join(ba[: 8].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            del ba[:8]
            freq = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            del ba[:32]
            sizes.append(size)
            freqs.append(freq)

        non_zero_size_freq = list(zip(sizes, freqs))
        Non_cdbk = huffman.codebook(non_zero_size_freq)
        rev_Non_cdbk = {value: key for (key, value) in Non_cdbk.items()}

        # Obtain a list of nonzero numbers
        non_zeros = deque()
        # print(" OBTAINING NONZERO LIST ...")
        for i in range(non_zero_length):
            # print(i)
            cursor = 1
            prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            while (prefix not in rev_Non_cdbk.keys()):
                cursor += 1
                prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            size = rev_Non_cdbk.get(prefix)
            del ba[:cursor]

            index = int("".join(ba[: size].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            amplitude = self.size_index2amp(size, index)
            non_zeros.append(amplitude)
            del ba[:size]
        # print(" OBTAINING NONZERO LIST, OK!")
        H_sym_seq_length = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
        del ba[: 32]
        H_sym_freq_length = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
        del ba[: 32]

        # Obtain codes list
        codes = []
        syms = []
        freqs = []

        for i in range(H_sym_freq_length):
            sym = chr(int("".join(ba[: 8].decode({'1': bitarray('1'), '0': bitarray('0')})), 2))
            del ba[:8]
            freq = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            del ba[:32]
            syms.append(sym)
            freqs.append(freq)
            
        sym_freq = list(zip(syms, freqs))
        Zt_cdbk = huffman.codebook(sym_freq)
        rev_Zt_cdbk = {value: key for (key, value) in Zt_cdbk.items()}

        # print(" OBTAINING CODES LIST ...")
        for i in range(H_sym_seq_length):
            cursor = 1
            prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            while (prefix not in rev_Zt_cdbk.keys()):
                cursor += 1
                prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            code = rev_Zt_cdbk.get(prefix)
            del ba[:cursor]
            codes.append(code)
        # print(" OBTAINING CODES LIST, OK!")

        if ba[:16] == NEXT_SYMBOL:
            # print(" READING BINARY FILE, FINISH!")
            pass
        else:
            raise ValueError("EOI marker mismatched.")
        rf.close()

        return coeffs, codes, non_zeros

    def get_init_coeffs(self) -> List[List[np.ndarray]]:
        """Get initial coefficients matrices.

        Returns:
            A list containing coefficient matrices corresponding to sub-band areas.

        """
        levels = self.T
        img = np.zeros((self.H, self.W), dtype= np.float32)
        (H, W) = img.shape

        h = H // 2 ** levels
        w = W // 2 ** levels

        coeffs = []
        for i in range(levels + 1):
            if i == 0:
                coeffs.append(img[: h, : w])
            else:
                coeffs.append([img[:h, w: 2 * w],
                               img[h:2 * h, w: 2 * w],
                               img[h: 2 * h, : w]])
                h *= 2
                w *= 2

        return coeffs

    def i_prediction(self, seq: List[List[float]]) -> List[List[float]]:
        """Inverse prediction.

        Args:
            seq: A sequence undergoing prediction.

        Returns:
            A sequence undergoing inverse prediction.

        """
        for i in range(0, len(seq) - 1):
            seq[i + 1] = seq[i] + seq[i + 1]
        return seq

    def size_index2amp(self,
                       size: int,
                       index: int) -> int:
        """Obtain corresponding `amplitude` by using `size` and `index`.

        Args:
            size: The exact bit length of binary form.
            index: Given specific `size`, `index` stands for the index of corresponding `amplitude`.

        Returns:
            An integer representing `amplitude` in Run-length Coding.

        """
        amp = -1
        if index < 2 ** (size - 1):
            amp = index - 2 ** size + 1
        else:
            amp = index
        return amp

    def saveDecodedFile(self) -> None:
        """Save decoded image."""
        try:
            path = ".\\client_rec\\" + self.name + ".png"
            cv2.imwrite(path, self.img)
        except Exception as e:
            print(e)

    def i_quantization(self, q_img: np.ndarray):
        """Inverse quantization.

        Args:
            q_img: An image undergoing quantization.

        Returns:
            An image recovered from quantization.
        """
        q_img = q_img * self.q
        return q_img

    def iDWT(self) -> np.ndarray:
        """Execute IDWT.

        Returns:
            Decoded image.

        """
        wtf = WaveletTransformer(self.dwt_img, self.T, (self.h, self.w), "s")
        img = wtf.iDWT()
        return img

