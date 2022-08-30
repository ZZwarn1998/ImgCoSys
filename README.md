# Intro

&emsp;This is a simple Image Compression System which has the function of network communication. I realized the final project in class, EEE5347 Image & Video Compression and Network Communication.  Here is a simple illustration roughly showing the overall structure of network communication process of the project. 

<p style="text-align: center;"> <img src=".\_readme_img\structure.png"></p>

# Usage

## Experiment

&emsp;If you want to have a look at the process of image compression and decompression, go to `main.py` and click ***run button*** to start. All data are save in folder `./data`.  In this section,  I wrote `plot.py` to show the pairwise relationships between some variables, such as R (Rate), D (Distortion), CR (Compressed Rate) and Q (Quantization step size). Assuming that the name of image is `image1`, before we draw plots by using it, you should **make sure** that you have copied a csv file named `image1.csv`, pasted it in folder , `./data/collect` , and assigned the parameter, `name`, `"image1"`, which is in `plot.py`.

## Network Communication

&emsp;If you want to use network communication, you should do like this. After you run `client.py` and `server.py`, you should designate **IP address** and **Port** of target server for `client.py`.  Then,   just follow the hint, type command line and you can use image compression system via network communication. Additionally, decoded image is saved in folder `./client_rec`. 

# Thanks

&emsp;Thanks for the author, [aparande](https://github.com/aparande), of the repository, [EZWImageCompression](https://github.com/aparande/EZWImageCompression). The repository inspires me how to realize zero-tree.
