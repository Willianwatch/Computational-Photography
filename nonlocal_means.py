import cv2
import numpy as np

def psnr(img1: np.ndarray, img2: np.ndarray):
    return 10*np.log(255*255.0/(((img1.astype(np.float32) - img2)**2).mean()))/np.log(10)

def double2uint8(img: np.ndarray, ratio=1.0):
    return np.clip(np.round(img*ratio), 0, 255).astype(np.uint8)

def make_kernel(f):
    """
    生成高斯核，越靠近中心位置的像素，权重越高
    """
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        kernel[f-d:f+d+1, f-d:f+d+1] += (1.0/((2*d+1)**2))
    return kernel/kernel.sum()

def nonlocal_means_filter(img: np.ndarray, h_=10, template_window_size=5,  search_window_size=11):
    f = template_window_size//2
    t = search_window_size//2
    height, width = img.shape[:2]
    pad_length = t+f
    img2 = np.pad(img, pad_length, 'symmetric')
    kernel = make_kernel(f)
    h = (h_**2)
    img_ = img2[pad_length-f:pad_length+f+height, pad_length-f:pad_length+f+width]

    average = np.zeros(img.shape)
    sweight = np.zeros(img.shape)
    wmax =  np.zeros(img.shape)
    for i in range(-t, t+1):
        for j in range(-t, t+1):
            if i==0 and j==0:
                continue
            I2_ = img2[pad_length+i-f:pad_length+i+f+height, pad_length+j-f:pad_length+j+f+width]
            # TODO: filter2D 的作用
            w = np.exp(-cv2.filter2D((I2_ - img_)**2, -1, kernel)/h)[f:f+height, f:f+width]
            sweight += w
            wmax = np.maximum(wmax, w)
            average += (w*I2_[f:f+height, f:f+width])
    # 原图像需要乘于最大权重参与计算
    average += (wmax*img)
    # sweight 为 weight 之和，用于计算 weight 的归一化
    sweight += wmax
    return average / sweight

if __name__ == '__main__':
    I = cv2.imread('yifei.png', flags=cv2.IMREAD_GRAYSCALE)
    sigma = 20.0
    I1 = double2uint8(I + np.random.randn(*I.shape) *sigma)
    print('噪声图像PSNR',psnr(I, I1))
    R1  = cv2.medianBlur(I1, 5)
    print('中值滤波PSNR',psnr(I, R1))
    R2 = cv2.fastNlMeansDenoising(I1, None, sigma, 5, 11)
    print('opencv的NLM算法',psnr(I, R2))
    R3 = double2uint8(nonlocal_means_filter(I1.astype(np.float32), sigma, 5, 11))
    print('NLM PSNR',psnr(I, R3))
    cv2.imwrite("./yifei_nlm_out.png", R3)
