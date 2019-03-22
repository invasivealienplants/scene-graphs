import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet101
import numpy as np
import scipy.misc as misc
import os
import json
import csv
import cv2
import math
import base64
import sys
import h5py
import utils
import torch.utils.model_zoo as model_zoo
csv.field_size_limit(sys.maxsize)

bboxes = h5py.File('/home/pp456/px2graph/exp/sg_results/coco_scenegraphs_px2graph.h5','r')
coco_image_data = json.load(open('/home/pp456/data/genome/driver/data/COCO/coco_image_data.json','r'))
object_types = utils.get_object_types()
predicate_types = utils.get_predicate_types()

info = (coco_image_data,bboxes,object_types,predicate_types)

trainfeaturefile = open('COCO_train_features.tsv','w+b')
valfeaturefile = open('COCO_val_features.tsv','w+b')

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

trainwriter = csv.DictWriter(trainfeaturefile, delimiter='\t', fieldnames=FIELDNAMES)
valwriter = csv.DictWriter(valfeaturefile, delimiter='\t', fieldnames=FIELDNAMES)

resnet = resnet101()
resnet.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'))
resnet.cuda()
resnet.eval()

def extract_feature_map(model, x):
    y = model.conv1(x)
    y = model.bn1(y)
    y = model.relu(y)
    y = model.maxpool(y)

    y = model.layer1(y)
    y = model.layer2(y)
    y = model.layer3(y)
    y = model.layer4(y)

    return y

for idx in range(len(coco_image_data)):

    item = {}

    region_features = []
    region_boxes = []

    if idx % 100 == 0:
        print(idx)

    image_id = coco_image_data[bboxes['idx'][idx]]['image_id']

    image_path = '/home/pp456/COCO/images/train2014/COCO_train2014_'+str(image_id).zfill(12)+'.jpg'
    train = True
    if not os.path.isfile(image_path):
        image_path = '/home/pp456/COCO/images/val2014/COCO_val2014_'+str(image_id).zfill(12)+'.jpg'
        train = False
    image = misc.imread(image_path)
    image = (image-np.mean(image,axis=(0,1)))
    image = image/np.std(image,axis=(0,1))

    if len(image.shape) == 2:
        continue
    h,w = image.shape[0],image.shape[1]
    if h < w:
        nh,nw = 600,int(float(w)*600/h)
    else:
        nh,nw = int(float(h)*600/w),600

    image = cv2.resize(image,(nw,nh))
    image = torch.FloatTensor(image).reshape(1,nh,nw,3).permute(0,3,1,2)
    image = image.cuda()
 
    image_feature = extract_feature_map(resnet, image)
    imf_h,imf_w = image_feature.size(2),image_feature.size(3)
    skip = False

    objs,rels = utils.get_graph_matrix(idx, info, object_threshold=0.3, only_connected=True, nonmax_suppress=0.7)

    for obj in objs:
        x0,y0,x1,y1 = obj[3:6]
        nx0,ny0 = int(float(x0)/w*imf_w),int(float(y0)/h*imf_h)
        nx1,ny1 = int(math.ceil(float(x1)/w*imf_w)),int(math.ceil(float(y1)/h*imf_h))
        region_crop = image_feature[:,:,ny0:ny1,nx0:nx1]
        if region_crop.size(2) == 0 or region_crop.size(3) == 0:
            skip = True
            break 

        # region_resize = F.adaptive_max_pool2d(region_crop,(7,7)) # adaptive max pool
        # region_feature = resnet.avgpool(region_resize)
        region_feature = torch.mean(region_crop,dim=-1) # mean pool
        region_feature = torch.mean(region_feature,dim=-1)
        region_feature = region_feature.view(-1)
        
        region_feature = region_feature.detach()
        region_feature = region_feature.cpu().numpy()

        region_features.append(region_feature.flatten())
        region_boxes.append(obj[3:6])

    if skip:
        continue

    item['features'] = base64.b64encode(region_features.reshape([-1]))
    item['image_w'] = w
    item['image_h'] = h
    item['num_boxes'] = len(objs)
    item['image_id'] = image_id
    item['boxes'] = base64.b64encoderegion_boxes.reshape([-1]))
    if train:
        trainwriter.writerow(item)
        trainwriter.flush()
    else:
        valwriter.writerow(item)
        valwriter.flush()

trainwriter.close()
valwriter.close()



    








