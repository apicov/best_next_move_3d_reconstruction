from scipy.spatial import cKDTree as KDTree
import numpy as np
import cv2

def chamfer_d(pc_1, pc_2):
    tree = KDTree(pc_1)
    ds, _ = tree.query(pc_2)
    d_21 = np.mean(ds)

    tree = KDTree(pc_2)
    ds, _ = tree.query(pc_1)
    d_12 = np.mean(ds)
    return d_21 + d_12

def get_mask(img):
   res=np.array(img)[:,:,:3]
   mask1=np.all((res==[70,70,70]), axis=-1) 
   mask2=np.all((res==[71,71,71]), axis=-1) 
   mask3=np.all((res==[72,72,72]), axis=-1)
   mask12=np.bitwise_or(mask1,mask2)
   mask_inv=np.bitwise_or(mask12,mask3)
   mask=np.bitwise_not(mask_inv)
   return 255*mask

def get_frame(img):
    img=np.array(img)[:,:,:3]
    mask1=np.all((img==[70,70,70]), axis=-1)
    mask2=np.all((img==[71,71,71]), axis=-1)
    mask3=np.all((img==[72,72,72]), axis=-1)
    mask12=np.bitwise_or(mask1,mask2)
    mask=np.bitwise_or(mask12,mask3) 
    res=cv2.bitwise_and(img,img,mask = (255-255*mask).astype(np.uint8))
    res_bg=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    small=cv2.resize(res_bg, (84,84))
    #return res
    return small
