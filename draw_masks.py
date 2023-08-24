import os
import mmcv
from mmdet.core.visualization.image import imshow_det_bboxes
import numpy as np
import pycocotools.mask as maskUtils
import argparse


def load_filename_with_extensions(data_path, filename):
    """
    Returns file with corresponding extension to json file.
    Raise error if such file is not found.

    Args:
        filename (str): Filename (without extension).

    Returns:
        filename with the right extension.
    """
    full_file_path = os.path.join(data_path, filename)
    # List of image file extensions to attempt
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    # Iterate through image file extensions and attempt to upload the file
    for ext in image_extensions:
        # Check if the file with current extension exists
        if os.path.exists(full_file_path + ext):
            return full_file_path + ext  # Return True if file is successfully uploaded
    raise FileNotFoundError(f"No such file {full_file_path}, checked for the following extensions {image_extensions}")

def draw_masks(filename, data_path, output_path):
    if (os.path.exists(os.path.join(output_path, 'sa_' + filename + '_semantic.png'))==0):
      img = mmcv.imread(load_filename_with_extensions(data_path, 'sa_' + filename))
      anns = mmcv.load(os.path.join(output_path, 'sa_' + filename+'_semantic.json'))
      bitmasks, class_names = [], []

      for ann in anns['annotations']:
          bitmasks.append(maskUtils.decode(ann['segmentation']))
          class_names.append(ann['class_name'])
      imshow_det_bboxes(img,
                  bboxes=None,
                  labels=np.arange(len(bitmasks)),
                  segms=np.stack(bitmasks),
                  class_names=class_names,
                  font_size=25,
                  show=False,
                  out_file=os.path.join(output_path, 'sa_' + filename+'_semantic.png'))

      # Delete variables that are no longer needed
      del img
      del anns
    
    else:
      print('Have been processed before: ', os.path.join(output_path, 'sa_' + filename + '_semantic.png'))

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Concatenation image and segmentation.')
parser.add_argument('--list_img', type=str, default='list_img.txt', help='Path to the input file containing id images with target categories.')
parser.add_argument('--data_path', type=str, default='sa_1b', help='Directory to store images.')
parser.add_argument('--output_path', type=str, default='output', help='Path to the output file containing annotated images.')
args = parser.parse_args()

# Read the file names and URLs
with open(args.list_img, 'r') as f:
  file_content = f.read()
  list_img = file_content.split("\n")

for i in range(len(list_img)):
  list_img[i] = list_img[i][:6]
if '' in list_img:
  list_img.remove('');

for id in list_img:
  draw_masks(id, args.data_path, args.output_path)