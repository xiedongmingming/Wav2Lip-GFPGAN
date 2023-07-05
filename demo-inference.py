basePath = "F:\\workspace\\github\\xiedongmingming\\Wav2Lip-GFPGAN\\Wav2Lip-GFPGAN"



wav2lipFolderName = 'Wav2Lip-master'
gfpganFolderName = 'GFPGAN-master'

wav2lipPath = basePath + '/' + wav2lipFolderName
gfpganPath = basePath + '/' + gfpganFolderName

!wget 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth' -O {wav2lipPath}'/face_detection/detection/sfd/s3fd.pth'

!gdown https://drive.google.com/uc?id=1fQtBSYEyuai9MjBOF8j7zZ4oQ9W2N64q --output {wav2lipPath}'/checkpoints/'

import os

outputPath = basePath+'/outputs'

inputAudioPath = basePath + '/inputs/kim_audio.mp3'
inputVideoPath = basePath + '/inputs/kimk_7s_raw.mp4'

lipSyncedOutputPath = basePath + '/outputs/result.mp4'

if not os.path.exists(outputPath):

  os.makedirs(outputPath)

!cd $wav2lipFolderName && python inference.py \
--checkpoint_path checkpoints/wav2lip.pth \
--face {inputVideoPath} \
--audio {inputAudioPath} \
--outfile {lipSyncedOutputPath}



import cv2

from tqdm import tqdm
from os import path

import os

inputVideoPath = outputPath+'/result.mp4'
unProcessedFramesFolderPath = outputPath+'/frames'

if not os.path.exists(unProcessedFramesFolderPath):

  os.makedirs(unProcessedFramesFolderPath)

vidcap = cv2.VideoCapture(inputVideoPath)

numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

fps = vidcap.get(cv2.CAP_PROP_FPS)

print("FPS: ", fps, "Frames: ", numberOfFrames)

for frameNumber in tqdm(range(numberOfFrames)):

    _,image = vidcap.read()

    cv2.imwrite(path.join(unProcessedFramesFolderPath, str(frameNumber).zfill(4)+'.jpg'), image)


!cd $gfpganFolderName && \
  python inference_gfpgan.py -i $unProcessedFramesFolderPath -o $outputPath -v 1.3 -s 2 --only_center_face --bg_upsampler None

import os

restoredFramesPath = outputPath + '/restored_imgs/'

processedVideoOutputPath = outputPath

dir_list = os.listdir(restoredFramesPath)
dir_list.sort()

import cv2
import numpy as np

batch = 0
batchSize = 300

from tqdm import tqdm

for i in tqdm(range(0, len(dir_list), batchSize)):

    img_array = []

    start, end = i, i + batchSize

    print("processing ", start, end)

    for filename in tqdm(dir_list[start:end]):

        filename = restoredFramesPath + filename;

        img = cv2.imread(filename)

        if img is None:
            continue

        height, width, layers = img.shape

        size = (width, height)

        img_array.append(img)

    out = cv2.VideoWriter(processedVideoOutputPath + '/batch_' + str(batch).zfill(4) + '.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    batch = batch + 1

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()


concatTextFilePath = outputPath + "/concat.txt"

concatTextFile=open(concatTextFilePath,"w")

for ips in range(batch):

  concatTextFile.write("file batch_" + str(ips).zfill(4) + ".avi\n")

concatTextFile.close()

concatedVideoOutputPath = outputPath + "/concated_output.avi"

!ffmpeg -y -f concat -i {concatTextFilePath} -c copy {concatedVideoOutputPath}

finalProcessedOuputVideo = processedVideoOutputPath+'/final_with_audio.avi'

!ffmpeg -y -i {concatedVideoOutputPath} -i {inputAudioPath} -map 0 -map 1:a -c:v copy -shortest {finalProcessedOuputVideo}

from google.colab import files

files.download(finalProcessedOuputVideo)