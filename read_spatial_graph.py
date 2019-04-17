# example for reading spatial graph, try python 2.7

import json
import base64
import numpy as np

spatial_graphs = json.load(open('coco_spatial_graph_andersen_490.json','r'))

idx = spatial_graphs.keys()[0]
example_graph = spatial_graphs[idx]

image_id = example_graph['image_id'] # COCO image id
image_width = example_graph['image_w'] # image width
image_height = example_graph['image_h'] # image height
num_boxes = example_graph['num_boxes'] # number of boxes in graph
num_edges = example_graph['num_edges'] # number of edges in graph
boxes = np.frombuffer(base64.decodestring(example_graph['boxes']),dtype=np.float32).reshape([int(num_boxes),-1])
# each row in boxes is [x0,y0,x1,y1]
edges = np.frombuffer(base64.decodestring(example_graph['edges']),dtype=np.int16).reshape([int(num_edges),-1])
# each row in edges is [i,j,label] : corresponding to edge (boxes[i],boxes[j],label)
