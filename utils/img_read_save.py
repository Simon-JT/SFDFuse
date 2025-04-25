import numpy as np
import cv2
import os
from skimage.io import imsave

def image_read_cv2(path, mode='RGB'):
    """Read and preprocess image
    Args:
        path: Image path
        mode: 'RGB', 'GRAY', 'YCrCb' or 'FREQ' (new frequency domain mode)
    Returns:
        Processed image array
    """
    img_BGR = cv2.imread(path).astype('float32')
    assert mode in ['RGB', 'GRAY', 'YCrCb', 'FREQ'], 'mode error'
    
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    elif mode == 'FREQ':
        # Special mode for frequency domain processing
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
        # Ensure image dimensions are powers of 2 (optimized for FFT)
        h, w = img.shape
        h2, w2 = 2**int(np.ceil(np.log2(h))), 2**int(np.ceil(np.log2(w)))
        if h != h2 or w != w2:
            img = cv2.resize(img, (w2, h2))
    
    return img

def img_save(image, imagename, savepath):
    """Save fused image
    Args:
        image: Image array to save
        imagename: Image name
        savepath: Save path
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # Ensure image values are in valid range
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    imsave(os.path.join(savepath, f"{imagename}.png"), image)

def prepare_freq_fusion(img):
    """Prepare image for frequency domain fusion
    Args:
        img: Input image
    Returns:
        Processed image suitable for frequency domain processing
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Normalize to [0,1] range
    img = img.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    img = img[np.newaxis, np.newaxis, ...]
    
    return img 