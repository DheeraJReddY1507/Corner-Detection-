from PIL import Image as im
import numpy as np

def convolution(img, kernel):
    (image_height,image_width)=img.shape[:2]
    (kernel_height,kernl_width)=kernel.shape[:2]
    pad_h=int(kernel.shape[0]//2)
    pad_w=int(kernel.shape[1]//2)
    output=np.zeros(img.shape)
    for i in range(pad_h,image_height-pad_h):
        for j in range(pad_w,image_width-pad_w):
            center=img[i - pad_h : i + pad_h + 1, j - pad_w : j + pad_w + 1]
            output[i,j]=(center*kernel).sum()
    return output

def gradient_x(img):
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return convolution(img,kernel_x)
def gradient_y(img):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return convolution(img,kernel_y)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def CornerDetection(data):
    imgs = imgh = data
    imgs = np.dstack((imgs,imgs,imgs))
    imgh = np.dstack((imgh,imgh,imgh))

    if len(data.shape)>2:
        data=rgb2gray(image)

    data=np.pad(data,1,mode='edge')

    data_x=gradient_x(data)
    data_y=gradient_y(data)

    Hmatrix=np.zeros(shape=(data.shape[0],data.shape[1]))
    Smatrix=np.zeros(shape=(data.shape[0],data.shape[1]))

    Scount=0
    Hcount=0
    Sthreshold=50000
    k=0.03  #Suggested k value in many papers.

    (data_height,data_width)=data.shape[:2]
    window_h=window_w=1
    output=np.zeros(data.shape)

    for i in range(window_h,data_height-window_h):
        for j in range(window_w,data_width-window_w):
            Ix=data_x[i - window_h : i + window_h + 1, j - window_w : j + window_w + 1]
            h11=Ix*Ix
            h11=h11.sum()
            Iy=data_y[i - window_h : i + window_h + 1, j - window_w : j + window_w + 1]
            h22=Iy*Iy
            h22=h22.sum()
            h12=Ix*Iy
            h12=h21=h12.sum()
            response=((h11*h22)-(h12*h21))-k*(h11+h22)
            D=(4*h12*h21)+pow((h11-h22),2)
            temp=[]
            temp.append(0.5*((h11+h22)+pow(D,0.5)))
            temp.append(0.5*((h11+h22)-pow(D,0.5)))
            r_min=min(temp)
            if r_min>Sthreshold:
                imgs[i-1][j-1][:] = [255, 0, 0]
                Scount+=1
            if response>6500000000:
                Hcount+=1
                imgh[i-1][j-1][:] = [255, 0, 0]
    im.fromarray(imgs).show()
    im.fromarray(imgh).show()
    print("Shi-Tomasi Corners : "+str(Scount)+" , "+"Harris Corners : "+str(Hcount))

im_1 = im.open(r"C:\Users\Dheeraj\Desktop\HW1_Q2\Image1.jpg")     #image file path
data_1=np.array(im_1)
CornerDetection(data_1)
