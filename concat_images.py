import cv2
import os
import argparse

def concat_save(list_img, img_path, semantic_path, output_path):
  for img_id in list_img:
    img = cv2.imread(os.path.join(img_path,'sa_'+img_id+'.jpg'))
    semantic = cv2.imread(os.path.join(semantic_path,'sa_'+img_id+'_semantic.png'))
    img_semantic = cv2.hconcat([img, semantic])
    cv2.imwrite(os.path.join(output_path,'sa_'+img_id+'_visual.png'), img_semantic)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Concatenation image and segmentation.')
parser.add_argument('--list_img', type=str, default='list_img.txt', help='Path to the input file containing id images with target categories.')
parser.add_argument('--img_path', type=str, default='sa_1b', help='Directory to store images.')
parser.add_argument('--semantic_path', type=str, default='output', help='Directory to store segmentation.')
parser.add_argument('--output_path', type=str, default='visualization', help='Path to the output file containing visualization.')
args = parser.parse_args()

# Read the file names and URLs
with open(args.list_img, 'r') as f:
  file_content = f.read()
  list_img = file_content.split("\n")

for i in range(len(list_img)):
  list_img[i] = list_img[i][:6]
if '' in list_img:
  list_img.remove('');

concat_save(list_img, args.img_path, args.semantic_path, args.output_path)