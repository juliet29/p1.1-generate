import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import pandas as pd

from sklearn.cluster import KMeans


import skimage.segmentation as seg
from skimage.segmentation import mark_boundaries
import skimage.measure as meas

from shapely import Polygon
import shapely.plotting as splt

from plan_reader import PlanReader
from graph_logic import GraphLogic
from visualizer import Visualizer

from helpers import *



class Processor:
    def __init__(self):
        self.v = Visualizer()


    def read_plan(self, PATH, DEBUG=False):
        reader = PlanReader(PATH)
        try:
            self.tensor = reader.image2tensor()
            self.tensor_labels = reader.segment_tensor()
            self.tensor_labels_w_doors  = reader.further_segment_doors()
            self.regions = reader.array2shapely()
            if DEBUG:
                fig = self.v.view_plan_shapely(self.regions)
        except:
            print("Reading plan failed")

        return 
    
    def configure_graph(self, DEBUG=False):
        graph_editor = GraphLogic(self.regions)
        self.CG = graph_editor.init_graph()
