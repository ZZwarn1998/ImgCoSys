import copy
import sys
import cv2
import numpy as np



class Debachuy_5_3:
    """Debachuies (5, 3) filter bank."""
    h_0 = [-1/8, 1/4, 3/4, 1/4, -1/8]
    h_1 = [-1/2, 1, -1/2]
    g_0 = [1/2, 1, 1/2]
    g_1 = [-1/8, -1/4, 3/4, -1/4, -1/8]


class WaveletTransformer:
    """The class is designed to use methods relative to discrete wavelet transformation(dwt).

    Attributes:
        mode (str): The mode of the object, "a" means analysis mode and "s" means synthesis mode.
        img (np.ndarray): Raw image.
        H (int): The height of raw image.
        W (int): The width of raw image.
        flt (Debachuy_5_3): Filter bank.
        T (int): The number of DWT or IDWT.
    """
    def __init__(self, img: np.ndarray, t: int, shape: tuple, mode: str="a"):
        """Constructor of class `WaveletTransformer`.

        Args:
            img (np.ndarray): Raw image.
            t (int):  The nunmber of DWT or IDWT.
            shape (tuple): A 2 dimensional tuple standing for the shape of original image or LL area in image after DWT.
            mode (str): A string which represents the mode of WaveletTransformer.
        """
        self.mode = mode
        (h, w) = shape
        self.img = copy.copy(img)
        self.H = h
        self.W = w
        self.flt = Debachuy_5_3
        self.T = t
        self.dwt_img = None
        self.idwt_img = None
        # self.rec = rec # know if the object is receiver? False or True.

    def modeCheck(self, cur_mode: str):
        """This method is used to check if current mode is proper.

        Args:
            cur_mode (str):  A string used for representing current mode.

        """
        if self.mode == cur_mode:
            return
        ex = Exception("    You have set up wrong mode! Please check your codes and switch to ANOTHER mode!")
        raise ex

    def DWT(self):
        """This method is used to carry on DWT.

        Returns:
            img (np.ndarray): An image undergoing DWT.

        """
        cur_mode = "a"
        try:
            self.modeCheck(cur_mode)
        except Exception as e:
            print(e)
            sys.exit(1)

        t = self.T
        r = self.H
        c = self.W
        img = copy.copy(self.img)

        img = img.astype(np.float32) / 255

        for i in range(t):
            for index, row in enumerate(img[0:r]):
                l = self.down_sampling(self.pass_filter(row[0:c], self.flt.h_0))
                h = self.down_sampling(self.pass_filter(row[0:c], self.flt.h_1), pos="e")
                img[index, 0:c] = np.hstack((l, h))
            img = img.T
            for index, row in enumerate(img[0:c]):
                l = self.down_sampling(self.pass_filter(row[0:r], self.flt.h_0))
                h = self.down_sampling(self.pass_filter(row[0:r], self.flt.h_1), pos="e")
                img[index, 0:r] = np.hstack((l, h))
            img = img.T

            r = r // 2
            c = c // 2

        img = img.astype(np.float32) * 255

        self.dwt_img = img

        return img


    def down_sampling(self, row: np.ndarray, pos: str="o"):
        """Down sampling.

        Args:
            row (np.ndarray): An array waiting to undergo down sampling.
            pos (str): A string used to determine the parity of the length of row.

        Returns:
            row (np.ndarray): An array whose length is half of the length of `row`.

        """
        if pos == "o":
            return row[::2]
        else:
            return row[1::2]

    def pass_filter(self, row: np.ndarray, flt: Debachuy_5_3):
        """The method is designed to carry on high-pass or low-pass filter.

        Args:
            row (np.ndarray): An array waiting to undergo high-pass or low pass.
            flt (Debachuy_5_3): An array standing for filter.

        Returns:
            new_row (np.ndarray): An array going through high pass or low pass.

        """
        temp = np.pad(row, (len(flt) // 2, len(flt) // 2), "reflect")
        new_row = np.convolve(temp, flt[::-1], "valid")
        return new_row

    def iDWT(self):
        """This method is used to carry on IDWT.

        Returns:
            img (np.ndarray): Reconstructed image.
        """
        cur_mode = "s"
        try:
            self.modeCheck(cur_mode)
        except Exception as e:
            print(e)

        t = self.T
        r = self.H
        c = self.W
        img = copy.copy(self.img)

        img = img.astype(np.float32) / 255

        for i in range(t):
            img = img.T
            for index, row in enumerate(img[0: 2 * c]):
                h = self.pass_filter(self.up_sampling(row[r: 2 * r], pos="e"), self.flt.g_1)
                l = self.pass_filter(self.up_sampling(row[0: r]), self.flt.g_0)
                img[index, 0: 2 * r] = l + h
            img = img.T
            for index, row in enumerate(img[0: 2 * r]):
                h = self.pass_filter(self.up_sampling(row[c: 2 * c], pos="e"), self.flt.g_1)
                l = self.pass_filter(self.up_sampling(row[0: c]), self.flt.g_0)
                img[index, 0: 2 * c] = l + h

            r = r * 2
            c = c * 2

        img[img < 0] = 0
        img[img > 1] = 1
        img = (img * 255).astype(np.uint8)

        self.idwt_img = img

        return img

    def up_sampling(self, row: np.ndarray, pos: str="o"):
        """The method is designed to undergo up sampling.

        Args:
            row (np.ndarray): An array waiting to undergo down sampling.
            pos (str):  A string used for determining the parity of the length of row.

        Returns:
            new_row (np.ndarray): An array after up sampling.

        """

        new_row = np.zeros(2 * len(row))

        if pos == "o":
            for i in range(len(row)):
                new_row[2 * i] = row[i]
        else:
            for i in range(len(row)):
                new_row[1 + 2 * i] = row[i]
        return new_row
