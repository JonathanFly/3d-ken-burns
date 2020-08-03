#You can edit this cell then click the play button to update the file. 
#!/usr/bin/env python
import torch
import torchvision

import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################

arguments_strIn = './images/doublestrike.jpg'
arguments_strOut = './videos/fullrez.mp4'
arguments_strFps = '30'
arguments_strBitRate = "12M"

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--fps' and strArgument != '': arguments_strFps = strArgument # allow FPS value selection from cmd line
	if strOption == '--bitrate' and strArgument != '': arguments_strBitRate = strArgument # allow FPS value selection from cmd line
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
	# end

	##########################################################

if __name__ == '__main__':
	npyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

	intWidth = npyImage.shape[1]
	intHeight = npyImage.shape[0]

	fltRatio = float(intWidth) / float(intHeight)

	#intWidth = min(int(1024 * fltRatio), 1024)
	#intHeight = min(int(1024 / fltRatio), 1024)

	#npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

	process_load(npyImage, {})

	objFrom = {
	'fltCenterU': intWidth / 2.0, #You can play with these but some settings will crash it or produce a bad video file
	'fltCenterV': intHeight / 2.0, #You can play with these but some settings will crash it or produce a bad video file
	'intCropWidth': int(math.floor(0.97 * intWidth)),
	'intCropHeight': int(math.floor(0.97 * intHeight))
	}

	objTo = process_autozoom({
    #'fltCenterU': intWidth / 2.0, #Not sure of effect
	#'fltCenterV': intHeight / 2.0, #Not sure of effect
	'fltShift': 100.0,
	#'fltShift': 10.0, # try this with large zooms
	'fltZoom': 1.25,
	#'fltZoom': 40.25, # 40x zoom
	'objFrom': objFrom
	})

	npyResult = process_kenburns({
	'fltSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
	#'fltSteps': numpy.linspace(0.0, 40.0, 800).tolist(), #example very large zoom, 800 frames
	#'fltSteps': numpy.linspace(0.0, 20.0, 275).tolist(), # Zoom x20 and 275 frames 
	'objFrom': objFrom,
	'objTo': objTo,
	'boolInpaint': True
	})

	moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:] ], fps=int(arguments_strFps)).write_videofile(arguments_strOut, bitrate=arguments_strBitRate)
