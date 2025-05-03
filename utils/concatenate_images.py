import cv2
import os
import numpy as np
import glob 
def concatenate_images_vertically(directory):
    # Get all images, sort them
    image_files = sorted([f for f in os.listdir(directory) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Read images
    images = [cv2.imread(os.path.join(directory, f)) for f in image_files]
    
    # Ensure same width
    width = images[0].shape[1]
    images = [cv2.resize(img, (width, img.shape[0])) if img.shape[1] != width else img 
              for img in images]
    
    # Concatenate vertically and return
    return np.vstack(images)

# Example usage
if __name__ == "__main__":
    results = []
    #paths = glob.glob('/*')
    paths = [
          './results256/lq',
          './results256/difface',
          './results256/psfrgan',
        './results256/gfpgan',
        './results256/codeformer',    
         './results256/restoreformer',    
          './results256/restoreformerPP', 
        './results256/hq'
        ]
    for path in paths:
        print(path)
        result = concatenate_images_vertically(path)
        results.append(result)
    final_result = np.hstack(results)
    cv2.imwrite("resultvh256.png", final_result)
