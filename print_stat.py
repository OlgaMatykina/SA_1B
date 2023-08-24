import os
import json
import collections
import pprint
import sys
import argparse

def choose_cat(output_dir, target):
  image_mask={};
  count_json = 0;
  for file in os.listdir(output_dir):
    if file.endswith('.json'):
      count_json+=1;
      with open(output_dir+'/'+file) as f:
        d = json.load(f)
        data = d['annotations']
        for i in data:
          for elem in target:
            if elem in i['class_name']:
              if d['image']['image_id'] not in image_mask.keys():
                image_mask[d['image']['image_id']] = {};
                if elem not in image_mask[d['image']['image_id']].keys():
                  image_mask[d['image']['image_id']][elem]=1;
                else:
                  image_mask[d['image']['image_id']][elem]+=1;
              else:
                if elem not in image_mask[d['image']['image_id']].keys():
                  image_mask[d['image']['image_id']][elem]=1;
                else:
                  image_mask[d['image']['image_id']][elem]+=1;
  return image_mask, count_json;

def print_stat(output_dir, image_mask, count_json):
  print("Всего изображений: ", count_json)
  print("Автоматически отобранных изображений: ", len(image_mask))
  count_mask=0;
  for i in image_mask:
    for j in image_mask[i]:
      count_mask+=image_mask[i][j]
  print("Автоматически отобранных масок: ", count_mask)

def cut_choosen_cat(image_mask):
  list_del = []
  for elem in image_mask:
    if len(image_mask[elem])==1:
      for key in image_mask[elem].keys():
        if image_mask[elem][key]==1:
          list_del.append(elem)
  for elem in list_del:
    del image_mask[elem]
  return image_mask

def stat_cat(target, image_mask):
  dict_tar = dict(zip(target, [0]*len(target)))
  for elem in image_mask:
    for key in image_mask[elem].keys():
      dict_tar[key]+=image_mask[elem][key]
  return dict_tar

def images_for_visual(vis_target, image_mask):
  for elem in image_mask:
    for tar in vis_target:
      if tar in image_mask[elem].keys():
        print(elem, image_mask[elem])
        break;
# target = ['hose', 'cable', 'puddle', 'road', 'lawn', 'grass', 'earth', 'dirt', 'sidewalk', 'sand', 'break', 'pit', 'garbage', 'trash', 'rubble', 'border', 'rope', 'tube','parking'];
# output_dir = '/content/output/content/Semantic-Segment-Anything/output'
# stat_file = 'stat.txt'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Statistics about masks.')
parser.add_argument('--target_file', type=str, default='target.txt', help='Path to the input file containing target categories.')
parser.add_argument('--vis_target_file', type=str, default='vis_target.txt', help='Path to the input file containing target categories for visualization.')
parser.add_argument('--output_dir', type=str, default='output', help='Directory to store processed files.')
parser.add_argument('--stat_file', type=str, default='stat.txt', help='Path to the output file containing statistics about masks.')
parser.add_argument('--cut_stat_file', type=str, default='cut_stat.txt', help='Path to the output file containing cut statistics about masks.')
parser.add_argument('--list_images', type=str, default='list_images.txt', help='Path to the output file containing list of images for visualization.')
args = parser.parse_args()

# Read the target categories
with open(args.target_file, 'r') as f:
  file_content = f.read()
  target = file_content.split(",")

orig_stdout = sys.stdout
f = open(args.stat_file, 'w')
sys.stdout = f
image_mask, count_json = choose_cat(args.output_dir, target)
print_stat(args.output_dir, image_mask, count_json)
pprint.pprint(image_mask)
dict_tar = stat_cat(target, image_mask)
pprint.pprint(dict_tar)
sys.stdout = orig_stdout
f.close()

orig_stdout = sys.stdout
f = open(args.cut_stat_file, 'w')
sys.stdout = f
cut_image_mask = cut_choosen_cat(image_mask)
print_stat(args.output_dir, cut_image_mask, count_json)
pprint.pprint(cut_image_mask)
cut_dict_tar = stat_cat(target, cut_image_mask)
pprint.pprint(cut_dict_tar)
sys.stdout = orig_stdout
f.close()

# Read the target categories
with open(args.vis_target_file, 'r') as f:
  file_content = f.read()
  vis_target = file_content.split(",")

orig_stdout = sys.stdout
f = open(args.list_images, 'w')
sys.stdout = f
images_for_visual(vis_target, cut_image_mask)
sys.stdout = orig_stdout
f.close()