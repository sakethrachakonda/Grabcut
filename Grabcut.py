
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
img = Image.open(r"D:\Data Science\Python\image.jpg")
img_np = np.array(img)

img_pil = Image.fromarray(img_np)
img_pil.show()


mask=np.zeros(img_np.shape[:2],np.uint8)

bgModel=np.zeros((1,65),np.float64)
fgModel=np.zeros((1,65),np.float64)

h,w=img_np.shape[:2]
#rect=(30,30,700,700)
rect = (100, 200, img_np.shape[1] - 200, img_np.shape[0] - 250) 
cv2.grabCut(img_np,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)

mask2=np.where((mask ==2)| (mask==0),0,1).astype('uint8')

segmented_img = img_np * mask2[:, :, np.newaxis] 

segmented_pil = Image.fromarray(segmented_img)
segmented_pil.show()


plt.subplot(1,2,1)
plt.title("Grabcut")
#plt.xticks([]),plt.yticks([])
plt.imshow(img_np)
plt.axis("off")


plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(cv2.imread("D:\\Data Science\\Python\\image.jpg"),cv2.COLOR_BGR2RGB))
plt.title("Original")
#plt.xticks([]),plt.yticks([])
plt.axis("off")

plt.show()










 
