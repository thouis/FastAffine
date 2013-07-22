import sys
sys.path.append('C:/Python27/lib/site-packages')
import cv2
import numpy as np
from fastaffine import remap

if __name__ == '__main__':
    srcim = cv2.imread(sys.argv[1], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    trans = [float(s) for s in sys.argv[2:12]]
    R = np.array([[trans[0], trans[1]],
                  [trans[2], trans[3]]])
    T = np.array([[trans[4]],
                  [trans[5]]])
    base_j = int(trans[6])
    base_i = int(trans[7])
    outw = int(trans[8]) - base_j
    outh = int(trans[9]) - base_i
    output = np.zeros((outh, outw), dtype=np.uint8)
    offset_i = srcim.shape[0] // 2
    offset_j = srcim.shape[1] // 2
    remap(srcim, output, R, T, False, base_i, base_j, offset_i, offset_j)
    cv2.imwrite(sys.argv[12], output)
