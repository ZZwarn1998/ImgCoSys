import os
import cv2
import socket
import hashlib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from imageDecoder import ImageDecoder


def readRawFile(path: str) -> np.ndarray:
    """Read raw image.

    Args:
        path: The Path of selected image.

    Returns:
        Raw image.
    Raises:
        Exception:
    """
    img = None
    try:
        img = np.fromfile(path, dtype="uint8")
        img = img.reshape(512, 512)
    except Exception as e:
        print(e)
    return img


def PSNR(raw, rec):
    """calculate PSNR"""
    return peak_signal_noise_ratio(raw, rec)


if __name__ == "__main__":
    # generate object of socket connection
    client = socket.socket()
    # if you want to connect to another server, please change the setting of ip_port shown below.
    print("Welcome, please type IP and Port.")
    ip_port = None

    while(True):
        ip = input(">>IP:")
        port = input(">>PORT:")
        alp_judge = True
        ip = ip.strip()
        port = port.strip()
        for num in ip.split('.'):
            if not num.isnumeric():
                alp_judge = False

        if len(ip.split('.')) == 4 and alp_judge and port.isnumeric():
            ip_port = (ip, int(port))
            break
        else:
            print(" WRONG FORMAT! Please check your input.")
            print()
    print("="*100)
    print("Wait to connect the server...")
    client.connect(ip_port)
    print("Server has been connected.")

    while True:
        print("-"*100)
        print("<1>Please FOLLOW the pattern to type commend: get <name> <q> <T>")
        print("<2>If you want to sign out, please type: stop")
        print("<3>`name` should be chosen from the set, [\"image1\", \"image2\", \"image3\", \"image4\", \"image5\"].")
        print("<4>`q` is quantization step size.")
        print("<5>`T` is the number of DWT or IDWT.")
        content = input(">>")
        if content == "stop":
            break

        if len(content) == 0:
            continue
        con_sp = content.split(" ")
        if len(content.split(" ")) != 4:
            continue
        _, filename, q, T = con_sp

        if filename not in ["image1", "image2", "image3", "image4", "image5"]:
            print("Wrong filename!")
            continue

        q = int(q)
        if q <= 0:
            print("Wrong value q!")
            continue

        T = int(T)
        if T <= 0:
            print("Wrong value T!")
            continue

        if content.startswith("get"):
            client.send(content.encode("utf-8"))

            # length of receiver buffer
            server_response = client.recv(1024)
            file_size = int(server_response.decode("utf-8"))
            print("-"*100)
            print("Size of received file: %d Bytes" % (file_size))

            # receive file
            client.send("Ready to receive ".encode("utf-8"))
            rec_filename = "rec_" + filename + ".bin"

            filepath = ".\\bin_rec\\" + rec_filename

            f = open(filepath, "wb")
            received_size = 0
            m = hashlib.md5()

            # receive file
            while received_size < file_size:
                size = 0
                if file_size - received_size > 1024:
                    size = 1024
                else:
                    size = file_size - received_size

                data = client.recv(size)  # 多次接收内容，接收大数据
                data_len = len(data)
                received_size += data_len
                print("\rPercentage of received file: " + str(int(received_size/file_size*100)) + "%", end='', flush=True)

                m.update(data)
                f.write(data)

            f.close()
            print()
            print("Actual size of received file:", received_size, "Bytes")  # decode

            # MD5 checksum
            md5_sever = client.recv(1024).decode("utf-8")
            md5_client = m.hexdigest()
            print("-"*100)
            print("Server md5:", md5_sever)
            print("Received file md5:", md5_client)
            if md5_sever == md5_client:
                print("MD5 checksum successes")
            else:
                print("MD5 checksum fails")
            print("-"*100)
            img_name = filename
            print("Start Decoding!")
            decoder = ImageDecoder(filepath, q, img_name)
            img = decoder.run()
            decoder.saveDecodedFile()
            print("Finish!")
            print("-"*100)
            print("Please see transmitted image", img_name + ".png in .\\out folder.")
            raw_pic_path = ".\\pic\\" + filename + ".512"
            raw_img = readRawFile(raw_pic_path)
            D = PSNR(raw_img, img)

            bf_size = os.path.getsize(raw_pic_path)
            af_size = os.path.getsize(filepath)

            print("PSNR:", D)
            print("CR(Size of Raw File / Size of Compressed File):",  bf_size / af_size)
            print("-" * 100)

            print("Start showing decoded image.")
            cv2.imshow("lena", img)
            cv2.waitKey(10000)
            cv2.destroyWindow("lena")
            print("Finish showing decoded image.")

            print("="*100)
            print()
    client.close()

