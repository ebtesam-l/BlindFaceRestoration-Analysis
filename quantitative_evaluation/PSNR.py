import torch
import piq
from skimage.io import imread
import glob
import os

import argparse


def calculate_images_PSNR(dirGT,dirEI, imageres):

    """This method calculate the PSNR value based on ground truth images and enhanced images. 

    Args:
        dirGT (str): directoy containing ground truth images
        dirEI (str): directory containing enhanced images
        imageres (int): size of images

    Returns:
       float: The PSNR value. 
    """ 

    # Read ground truth images
    path_a = glob.glob(os.path.join(dirGT, "*.*"))
    count_a = len(path_a)
    path_b = glob.glob(os.path.join(dirEI, "*.*"))
    count_b = len(path_b)
    if(count_a != count_b):
        print("folders have different number of images, cannot compute PSNR")
        print(count_a)
        print(count_b)
        exit()

    GT = torch.zeros((count_a, 3, imageres,imageres))
    DI = torch.zeros((count_b, 3, imageres,imageres))
    i = 0
    for file in path_a:
        thisfile = file.split("/")[-1]
        GT[i,:,:,:] = torch.tensor(imread(file)).permute(2, 0, 1)[None, ...] / 255.
#        while( thisfile[0] == '0'):  #Leading zeros removal from filename
#            thisfile = thisfile[1:]
        file_enhanced = os.path.join(dirEI, thisfile)
        if(not os.path.exists(file_enhanced)):
            print("file: " + file + "exists in the ground truth but has no corresponding enhanced image, cannot compute PSNR")
            exit()
        DI[i,:,:,:] = torch.tensor(imread(file_enhanced)).permute(2, 0, 1)[None, ...] / 255.
        i +=1


    if torch.cuda.is_available():
    # Move to GPU to make computaions faster
        GT = GT#.cuda()
        DI = DI#.cuda()


    # To compute PSNR as a measure, use lower case function from the library.
    psnr_index = piq.psnr(GT,  DI, data_range=1., reduction='none')

    meanpsnr = psnr_index.mean()
    # The higher the PSNR, the better the quality of the compressed, or reconstructed image.
    return meanpsnr.cpu().numpy()

   

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument( "--dirGT", help="directory containing ground truth images, example: celeba_train/", required = True)
    argParser.add_argument( "--dirEI", help="directory containing enhanced images, example: enhanced/", required = True)
    argParser.add_argument( "--imagesize", help="resolution in pixels of images, example: 256, 512, 1024", required = True, type=int)

    args = argParser.parse_args()

    #Directory of ground truth images 
    dirGT = args.dirGT
    dirEI = args.dirEI
    imageres = args.imagesize

    #Directory of enhanced images
    print(calculate_images_PSNR(dirGT,dirEI, imageres))
