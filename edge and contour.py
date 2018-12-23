import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from PIL import Image

jpeg = cv2.imread('jpeg.jpg',0)

jpegBlur = cv2.GaussianBlur(jpeg,(3,3),0)

#prewitt operator
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(jpeg, -1, kernelx)
img_prewitty = cv2.filter2D(jpeg, -1, kernely)
img_prewitt = img_prewitty + img_prewittx

#sobel operator
img_sobelx = cv2.Sobel(jpeg,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(jpeg,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely

#robert operator
roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

roberts_cross_h = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )
vertical = ndimage.convolve( np.asarray( jpeg, dtype="int32" ), roberts_cross_v )
horizontal = ndimage.convolve( np.asarray( jpeg, dtype="int32" ), roberts_cross_h )
output_robert = np.sqrt( np.square(horizontal) + np.square(vertical))
output_robert = Image.fromarray( np.asarray( np.clip(output_robert,0,255), dtype="uint8"), "L" )

#canny operator
edge_canny = cv2.Canny(jpegBlur, 70, 100)

#laplacian
laplacian = cv2.Laplacian(jpegBlur,cv2.CV_64F)
abs_dst = cv2.convertScaleAbs(laplacian)

#show result
plt.subplot(2,3,1), plt.imshow(jpeg,"gray"), plt.title('Original')
plt.subplot(2,3,2), plt.imshow(img_prewitt,'gray'), plt.title('Prewitt')
plt.subplot(2,3,3), plt.imshow(img_sobel,'gray'), plt.title('Sobel')
plt.subplot(2,3,4), plt.imshow(output_robert,'gray'), plt.title('Roberts')
plt.subplot(2,3,5), plt.imshow(edge_canny,'gray'), plt.title('Canny')
plt.subplot(2,3,6), plt.imshow(abs_dst,'gray'), plt.title('Laplacian')
plt.savefig('edge and contour.jpg')
plt.show()



