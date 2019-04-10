import base64
import numpy as np
import csv
import sys
import os
import math
import json

coco_image_data = json.load(open('coco_image_data.json','r'))

csv.field_size_limit(sys.maxsize)
infiles = ['trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv',
          'trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',\
          'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0', \
           'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1']

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

graph_idx = {}

def get_iou(bb1, bb2):
    """
    calculates IOU, bbs have format [x0,y0,x1,y1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

for infile in infiles:

    print("reading " + str(infile))

    reader = csv.DictReader(open(infile,'r+b'), delimiter='\t', fieldnames = FIELDNAMES)

    for item in reader:
        new_item = {}
        new_item['image_id'] = int(item['image_id'])
        new_item['image_h'] = int(item['image_h'])
        new_item['image_w'] = int(item['image_w'])
        new_item['num_boxes'] = int(item['num_boxes'])
        new_item['boxes'] = item['boxes']

        edges = []
        boxes = np.frombuffer(base64.decodestring(item['boxes']), dtype=np.float32).reshape((item['num_boxes'],-1))
        diag = math.sqrt(new_item['image_h']**2.0+new_item['image_w']**2.0)
        for i,box_i in enumerate(boxes):
            for j,box_j in enumerate(boxes):
                x0,y0,x1,y1 = box_i
                a0,b0,a1,b1 = box_j
                if i != j:
                    if x0 >= a0 and x1 <= a1 and y0 >= b0 and y1 <= b1:
                        edges.append((i,j,0))
                    elif x0 <= a0 and x1 >= a1 and y0 <= b0 and y1 >= b1:
                        edges.append((i,j,1))
                    else:
                        iou = get_iou(box_i,box_j)
                        if iou >= 0.5:
                            edges.append((i,j,2))
                        else:
                            cx_i,cy_i = (x0+x1)/2,(y0+y1)/2
                            cx_j,cy_j = (a0+a1)/2,(b0+b1)/2
                            d_ij = math.sqrt(math.fabs(cx_j-cx_i)**2.0+math.fabs(cy_j-cy_i)**2.0)
                            if d_ij/diag > 0.5:
                                angle = math.atan2(cy_j-cy_i,cx_j-cx_i)
                                label = 4+min(int((angle+math.pi)/(math.pi/4)),7)
                                edges.append((i,j,label))
        new_item['num_edges'] = len(edges)
        new_item['edges'] = base64.b64encode(np.array(edges,dtype=np.int16))

        graph_idx[new_item['image_id']] = new_item

json.dump(graph_idx,open('coco_spatial_graph_andersen.json','w'))


        

    

