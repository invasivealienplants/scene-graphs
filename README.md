# scene-graphs

Repo for extracting + visualizing scene-graphs on MS-COCO train+val 2014.

Scene graphs are pre-extracted from the [px2graph repo](https://github.com/princeton-vl/px2graph), with Faster R-CNN proposals from [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/object-recognition-and-scene-understanding/how-good-are-detection-proposals-really/)

## Setup

Dependencies: cv2, matplotlib, scipy, json, urllib, h5py, graphviz (recommend installing graphviz with conda).

Data:

Install COCO image metadata from [here](), and place in source directory.

Install pre-extracted scene-graph data from [here](), and place in exp/sg_results/ - about 7 GB.

## Usage

All functions are in util.py, a demo is in visualize.ipynb. Functions get_graph_matrix() returns objects and predicates matrices under pruning specification (e.g. score thresholds, removing isolated nodes), and visualize() displays bounding boxes on image + scene graph visualization (requires graphviz). **NOTE:** change graphviz path in utils.py (should appear if installed with conda)

## Example

Example image:

![alt text](https://github.com/invasivealienplants/scene-graphs/blob/master/sample_images/image.png)

Corresponding graph (score threshold = 0.1, removed isolated nodes):

![alt text](https://github.com/invasivealienplants/scene-graphs/blob/master/sample_images/connected_graph.png)

Corresponding graph, including isolated nodes:

![alt text](https://github.com/invasivealienplants/scene-graphs/blob/master/sample_images/extra_graph.png)
