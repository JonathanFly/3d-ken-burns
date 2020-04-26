# Colab Link [Colab-Ken-Burns-High-Resolution](https://github.com/JonathanFly/3d-ken-burns/blob/master/3D_Ken_Burns_HighResolution.ipynb)

# This is the pressreset Colab with a couple of bug fixes and some extra information. I only updated fullrez.py

# Multi-FPS-Full-Rez-3d-Ken-Burns)
This is a fork of the original from sniklaus. It takes an image and generates a parallaxed video from the image. The original readme is below. 2 new scripts have been created. multi.py and fullrez.py. Both allow for you to select the output FPS. 

## fullrez.py usage
fullrez will process the image at full resolution. It takes the same flags as autozoom.py. You can process HD (1920x1080) and 2k (3840x2160) images on a 16GB card, and possibly a 11GB card. 4k is out of the question.
### WARNING: THIS TAKES A TON OF GPU MEMORY UP TO 10GB+ FOR LARGE IMAGES
```
python fullrez.py --fps 240 --in ./images/doublestrike.jpg --out ./autozoom.mp4
```
## multi.py
multi will take a directory of images and then output them as movies with the same file names as the originals. This processes them at the original resolution. It will then zip the files when complete to a zip file named download.zip. You can also bypass the zip file creation if you wish using the --nozip flag set to Y. It resizes the images the same way as the original. You MUST use a trailing slash on the directory names. There is no sanity check for it. If you run a ton of images at once, you risk running out of GPU memory. However, I've processed up to 8 without issue on 16GB GPU memory.
```
python multi.py --fps 120 --indir ./images/ --outdir ./videos/ --zipname myzipfile.zip --nozip Y
```

## autozoom.py
Standard autozoom.py has been altered to accept the fps flag as it outputs 25fps by default.
```
python autozoom.py --fps 24 --in ./images/doublestrike.jpg --out ./autozoom.mp4
```

# 3d-ken-burns
This is a reference implementation of 3D Ken Burns Effect from a Single Image [1] using PyTorch. Given a single input image, it animates this still image with a virtual camera scan and zoom subject to motion parallax. Should you be making use of our work, please cite our paper [1].

<a href="https://arxiv.org/abs/1909.05483" rel="Paper"><img src="http://content.sniklaus.com/kenburns/paper.jpg" alt="Paper" width="100%"></a>

## setup
To download the pre-trained models, run `bash download.bash`.

Several functions are implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository. Please also make sure to have the `CUDA_HOME` environment variable configured.

In order to generate the video results, please also make sure to have `pip install moviepy` installed.

## usage
To run it on an image and generate the 3D Ken Burns effect fully automatically, use the following command.

```
python autozoom.py --in ./images/doublestrike.jpg --out ./autozoom.mp4
```

To start the interface that allows you to manually adjust the camera path, use the following command. You can then navigate to `http://localhost:8080/` and load an image using the button on the bottom right corner. Please be patient when loading an image and saving the result, there is a bit of background processing going on.

```
python interface.py
```

To run the depth estimation to obtain the raw depth estimate, use the following command. Please note that this script does not perform the depth adjustment, see [#22](https://github.com/sniklaus/3d-ken-burns/issues/22) for information on how to add it.

```
python depthestim.py --in ./images/doublestrike.jpg --out ./depthestim.npy
```

To benchmark the depth estimation, run `python benchmark.py`. You can use it to easily verify that the provided implementation runs as expected.

## colab
If you do not have a suitable environment to run this projects then you could give Colab a try. It allows you to run the project in the cloud, free of charge. There are several people who provide Colab notebooks that should get you started. A few that I am aware of include one from [Arnaldo Gabriel](https://colab.research.google.com/github/agmm/colab-3d-ken-burns/blob/master/automatic-3d-ken-burns.ipynb), one from [Vlad Alex](https://towardsdatascience.com/very-spatial-507aa847179d), and one from [Ahmed Harmouche](https://github.com/wpmed92/3d-ken-burns-colab).

## video
<a href="http://content.sniklaus.com/kenburns/video.mp4" rel="Video"><img src="http://content.sniklaus.com/kenburns/video.jpg" alt="Video" width="100%"></a>

## license
This is a project by Adobe Research. It is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) and may only be used for non-commercial purposes. Please see the LICENSE file for more information.

## references
```
[1]  @article{Niklaus_TOG_2019,
         author = {Simon Niklaus and Long Mai and Jimei Yang and Feng Liu},
         title = {3D Ken Burns Effect from a Single Image},
         journal = {ACM Transactions on Graphics},
         volume = {38},
         number = {6},
         pages = {184:1--184:15},
         year = {2019}
     }
```

## acknowledgment
The video above uses materials under a Creative Common license or with the owner's permission, as detailed at the end.
