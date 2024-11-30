from PIL import Image, ImageStat
import cv2
import numpy as np


im = Image.open("Pepper.PNG")


print(im.format, im.size, im.mode)
# im.show()

stats = ImageStat.Stat(im)

for band,name in enumerate(im.getbands()):
    print(f'Band: {name}, min/max: {stats.extrema[band]}, stddev: {stats.stddev[band]}')


img = cv2.imread("Pepper.PNG")
Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
min = np.float32(np.min(Y))
max = np.float32(np.max(Y))
# compute contrast
contrast = (max-min)/(max+min)
print(min,max,contrast, img.mean())




img = cv2.imread("Miko2.jpg")
Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
min = np.float32(np.min(Y))
max = np.float32(np.max(Y))
# compute contrast
contrast = (max-min)/(max+min)
print(min,max,contrast, img.mean())


# Detect blobs
keypoints = detector.detect('Miko2.jpg')
