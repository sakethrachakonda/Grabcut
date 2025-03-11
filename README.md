import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
img = Image.open(r"D:\Data Science\Python\image.jpg")# adding a path location
img_np = np.array(img)

img_pil = Image.fromarray(img_np)#converting image into array
img_pil.show()

mask=np.zeros(img_np.shape[:2],np.uint8)

bgModel=np.zeros((1,65),np.float64)#background
fgModel=np.zeros((1,65),np.float64)#foreground

h,w=img_np.shape[:2]#Height,Width 
rect = (100, 200, img_np.shape[1] - 200, img_np.shape[0] - 250)#rectangle box size with x,y,H,W
cv2.grabCut(img_np,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)# initializing rectangle for GrabCut method
mask2=np.where((mask ==2)| (mask==0),0,1).astype('uint8')#pixels are converted into numeric

segmented_img = img_np * mask2[:, :, np.newaxis] #Extra dimension 
segmented_pil = Image.fromarray(segmented_img)
segmented_pil.show()

plt.subplot(1,2,1)
plt.title("Grabcut")
plt.imshow(img_np)
plt.axis("off")#reomove X,Y-axis to look clearly

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(cv2.imread("D:\\Data Science\\Python\\image.jpg"),cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off"))#reomove X,Y-axis to look clearly
plt.show()



