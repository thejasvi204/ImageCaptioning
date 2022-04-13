from PIL import Image

import json
import matplotlib.pyplot as plt

def display_images(img_list, cmap='gray', cols = 5, fig_size = (12, 12) ):
    """
    Display images in img_list
    """
    
    i = 1  # for subplot
    num_images = len(img_list)
    num_rows = num_images / cols
    plt.figure(figsize=fig_size)       
    
    
    for image in img_list:
        image_title = image['title']
        plt.subplot(num_rows, cols, i)        
        plt.title(image_title, fontsize=7)
        plt.imshow(image['image'], interpolation='nearest', aspect='auto')
        i += 1     
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.5)
    # plt.show()
    plt.tight_layout()
    plt.savefig('captions-50-out.jpg')
    
imgdata = []

jf = open('vis/vis.json')
vis = json.load(jf)

for i in vis:
    imgdata.append({'image':Image.open('../'+i['file_name']), 'title': i['caption']})

display_images(imgdata)