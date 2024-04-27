import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
from io import BytesIO

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def convertToJpeg(im,quality):
    with BytesIO() as f:
        im.save(f, format='JPEG',quality=quality)
        f.seek(0)
        return Image.open(f).convert('RGB')

def synthesize_blur(image):
    
    x=np.array(image)
    kernel_size_candidate=[(3,3),(5,5),(7,7)]
    kernel_size=random.sample(kernel_size_candidate,1)[0]
    std=random.uniform(1.,5.)

    #print("The gaussian kernel size: (%d,%d) std: %.2f"%(kernel_size[0],kernel_size[1],std))
    blur=cv2.GaussianBlur(x,kernel_size,std)

    return Image.fromarray(blur.astype(np.uint8))

def synthesize_gaussian(image,std_l,std_r):

    img_pil=pil_to_np(image)

    mean=0
    std=random.uniform(std_l/255.,std_r/255.)
    gauss=np.random.normal(loc=mean,scale=std,size=img_pil.shape)
    noisy=img_pil+gauss
    noisy=np.clip(noisy,0,1).astype(np.float32)

    return np_to_pil(noisy)

def synthesize_speckle(image,std_l,std_r):

    img_pil=pil_to_np(image)

    mean=0
    std=random.uniform(std_l/255.,std_r/255.)
    gauss=np.random.normal(loc=mean,scale=std,size=img_pil.shape)
    noisy=img_pil+gauss*img_pil
    noisy=np.clip(noisy,0,1).astype(np.float32)

    return np_to_pil(noisy)

def synthesize_salt_pepper(image,amount,salt_vs_pepper):

    img_pil=pil_to_np(image)

    out = img_pil.copy()
    p = amount
    q = salt_vs_pepper
    flipped = np.random.choice([True, False], size=img_pil.shape,
                               p=[p, 1 - p])
    salted = np.random.choice([True, False], size=img_pil.shape,
                              p=[q, 1 - q])
    peppered = ~salted
    out[flipped & salted] = 1
    out[flipped & peppered] = 0.
    noisy = np.clip(out, 0, 1).astype(np.float32)


    return np_to_pil(noisy)

def synthesize_low_resolution(img):
    w,h=img.size

    new_w=random.randint(int(w/2),w)
    new_h=random.randint(int(h/2),h)

    img=img.resize((new_w,new_h),Image.BICUBIC)

    if random.uniform(0,1)<0.5:
        img=img.resize((w,h),Image.NEAREST)
    else:
        img = img.resize((w, h), Image.BILINEAR)

    return img

def add_holes(image, num_holes, max_hole_size):
    width, height = image.size
    
    # Create a copy of the original image
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    
    for _ in range(num_holes):
        # Randomly select a position for the hole
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        
        # Randomly select the size of the hole
        hole_size = random.randint(1, max_hole_size)
        
        # Draw a dark circular spot (hole) on the image
        draw.ellipse([(x - hole_size, y - hole_size), (x + hole_size, y + hole_size)], fill=(0, 0, 0))
    
    return new_image

def synthesize_image(img):

    task_id=np.random.permutation(4)

    for x in task_id:
        if x==0 and random.uniform(0,1)<0.7:
            img = synthesize_blur(img)
        if x==1 and random.uniform(0,1)<0.7:
            flag = random.choice([1, 2, 3])
            if flag == 1:
                img = synthesize_gaussian(img, 5, 50)
            if flag == 2:
                img = synthesize_speckle(img, 5, 50)
            if flag == 3:
                img = synthesize_salt_pepper(img, random.uniform(0, 0.01), random.uniform(0.3, 0.8))
        if x==2 and random.uniform(0,1)<0.7:
            img=synthesize_low_resolution(img)
        if x==3 and random.uniform(0,1)<0.7:
            img=convertToJpeg(img,random.randint(40,100))

    img = add_holes(img, 10, 5)
    
    return img