import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.misc as misc
import json
import urllib.request
import h5py
from graphviz import Digraph

def get_image_data():
    return json.load(open('coco_image_data.json','r'))

# maps idx -> object type (name)
def get_object_types():
    objects = {}
    for i,line in enumerate(open('sorted_objects.txt','r')):
        line = line[:-1]
        objects[i] = line
    return objects

# maps idx -> predicate type (name)
def get_predicate_types():
    predicates = {}
    for i,line in enumerate(open('sorted_predicates.txt','r')):
        line = line[:-1]
        predicates[i] = line
    return predicates

'''
Returns tuple (objs, rels) for image image_data[idx] according to pruning specifications

Format: 
objs : n_objs x 7 matrix
    objs[:,0] : object confidence scores
    objs[:,1] : object class type (int), according to sorted_objects.txt
    objs[:,2] : object class score (pre-softmax, kind of irrelevant if we don't consider other classes)
    objs[:,3:6] : bounding box in original image, as [x0,y0,x1,y1]
    
rels : n_rels x 4 matrix
    rels[:,0] : rel confidence score
    rels[:,1] : subject index (i.e. objs[rel[:,1]] gives list of subjects)
    rels[:,2] : object index (i.e. objs[rel[:,2]] gives list of objects)
    rels[:,3] : predicate type (int), according to sorted_predicates.txt
    an edge is given as (subject-predicate-object), eg. man-has-hat
    
object_threshold determines cutoff score for object confidence - only objects with confidence score > object_threshold are included
rel_threshold determines score for rel confidence
info : (image_data,graphs,objects,predicates)
only_connected (slight misnomer)
    when True will only return nodes that have degree > 0
    else includes isolated nodes
'''
def get_graph_matrix(idx, info, object_threshold=0.3, rel_threshold=0.0, only_connected=False):
    image_data,graphs,objects,predicates = info
    
    image = image_data[graphs['idx'][idx]]
    
    n_objs = int(graphs['n_objs'][idx])
    n_rels = int(graphs['n_rels'][idx])
    
    objs = graphs['objs'][idx][:n_objs]
    rels = graphs['rels'][idx][:n_rels]
    
    # add nodes to graph
    objects_idx = set([])
    for idx,obj in enumerate(objs):
        obj_score = obj[0]
        if obj_score > object_threshold:
            object_class_idx = int(obj[1])
            objects_idx.add(idx)

    # add edges to graph
    edges_to_add = set([])
    connected_nodes = set([])
    order = np.flip(np.argsort(rels[:,0]),0)
    rels = rels[order]
    contained_rels = set([])
    for idx,rel in enumerate(rels):
        if rel[0] > rel_threshold:
            sbj_idx = int(rel[1])
            obj_idx = int(rel[2])
            if sbj_idx in objects_idx and obj_idx in objects_idx:
                if sbj_idx == obj_idx: # remove self loops
                    continue
                if not (sbj_idx,obj_idx) in contained_rels:
                    edges_to_add.add(idx)
                    connected_nodes.add(sbj_idx)
                    connected_nodes.add(obj_idx)
                    contained_rels.add((sbj_idx,obj_idx))
                    
    object_map = {}
    included_objects = []
    included_rels = []
    if only_connected:
        use_objects = connected_nodes
    else:
        use_objects = objects_idx
    for new_idx,old_idx in enumerate(list(use_objects)):
        included_objects.append(objs[old_idx])
        object_map[old_idx] = new_idx
    for edge_idx in list(edges_to_add):
        edge = rels[edge_idx]
        edge[1] = object_map[edge[1]]
        edge[2] = object_map[edge[2]]
        included_rels.append(edge)
        
    included_objects = np.array(included_objects)
    included_rels = np.array(included_rels)
    
    return included_objects,included_rels

# visualizes coco_image_data[idx] (not coco id)
# takes top predicate between two objects
# no self loops, multi-edges
# object, rel thresholds determine what score cutoff to use in pruning
# currently NOT pruning nodes without edges
def visualize(idx, info, object_threshold=0.3, rel_threshold=0.0, only_connected=False):
    
    image_data,graphs,objects,predicates = info

    image = image_data[graphs['idx'][idx]]

    urllib.request.urlretrieve(image['url'],"sample_images/" + image['url'].split("/")[-1])
    im = misc.imread("sample_images/" + image['url'].split("/")[-1])

    fig,ax = plt.subplots(figsize=(int(im.shape[0]/50),int(im.shape[1]/50)))

    h,w = im.shape[0],im.shape[1]
    ax.imshow(im)
    
    objs,rels = get_graph_matrix(idx,info,object_threshold,rel_threshold,only_connected)
    
    for object_instance in objs:
        rect = patches.Rectangle((int(object_instance[3]),int(object_instance[4])),int(object_instance[5]-object_instance[3]),int(object_instance[6]-object_instance[4]),linewidth=2,edgecolor='r',facecolor='none')
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width()/2.0
        cy = ry + rect.get_height()/2.0
        ax.annotate(objects[int(object_instance[1])], (cx, cy), color='w', weight='bold', 
                        fontsize=12, ha='center', va='center')
        ax.add_patch(rect)

    plt.show()

    g = Digraph()
    
    for idx,obj in enumerate(objs):
        object_class_idx = int(obj[1])
        g.node(str(objects[object_class_idx])+","+str(idx))
        
    for rel in rels:
        sbj_idx = int(rel[1])
        obj_idx = int(rel[2])
        sbj_class_idx = int(objs[sbj_idx][1])
        obj_class_idx = int(objs[obj_idx][1])
        node1 = str(objects[sbj_class_idx])+","+str(sbj_idx)
        node2 = str(objects[obj_class_idx])+","+str(obj_idx)
        g.edge(node1,node2,label=predicates[int(rel[3])])

#     objects_idx = {}

#     # add nodes to graph
#     nodes_to_add = set([])
#     for idx,obj in enumerate(graphs['objs'][idx][:n_objs]):
#         obj_score = obj[0]
#         if obj_score > object_threshold:
#             object_class_idx = int(obj[1])
#             nodes_to_add.add(str(objects[object_class_idx])+","+str(idx))
#             objects_idx[idx] = obj

#     # add edges to graph
#     rels = set([])
#     edges_to_add = set([])
#     connected_nodes = set([])
#     for rel in graphs['rels'][idx][:n_rels]:
#         if rel[0] > rel_threshold:
#             sbj_idx = int(rel[1])
#             obj_idx = int(rel[2])
#             if sbj_idx in objects_idx and obj_idx in objects_idx:
#                 if sbj_idx == obj_idx: # remove self loops
#                     continue
#                 sbj_class_idx = int(objects_idx[sbj_idx][1])
#                 obj_class_idx = int(objects_idx[obj_idx][1])
#                 o1 = objects[sbj_class_idx]
#                 o2 = objects[obj_class_idx]
#                 if not (sbj_class_idx,obj_class_idx) in rels:
#                     node1 = str(o1)+","+str(sbj_idx)
#                     node2 = str(o2)+","+str(obj_idx)
#                     edges_to_add.add(((node1,node2),predicates[int(rel[3])]))
#                     connected_nodes.add(node1)
#                     connected_nodes.add(node2)
#                     rels.add((sbj_class_idx,obj_class_idx))
                    
#     if only_connected:
#         for node in list(connected_nodes):
#             g.node(node)
#     else:
#         for node in list(nodes_to_add):
#             g.node(node)
        
#     for ((node1,node2),label) in list(edges_to_add):
#         g.edge(node1,node2,label=label)

    return g