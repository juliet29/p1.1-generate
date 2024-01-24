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

import traceback



class Processor:
    def __init__(self):
        self.v = Visualizer()


    def read_plan(self, PATH, num_doors=6, DEBUG=False):
        reader = PlanReader(PATH)
        try:
            self.tensor = reader.image2tensor()
            self.tensor_labels = reader.segment_tensor()
            self.tensor_labels_w_doors  = reader.further_segment_doors(num_doors)
            self.regions = reader.array2shapely()
            if DEBUG:
                fig = self.v.view_plan_shapely(self.regions)
        except:
            print("Reading plan failed")

        return 
    
    def configure_graph(self, DEBUG=False):
        graph_editor = GraphLogic(self.regions)
        try:
            self.CG = graph_editor.init_graph()
            self.door_adj_labels = graph_editor.study_graph_adj()
            self.correct_edges = graph_editor.create_correct_adj()
            self.CG = graph_editor.create_correct_graph()
            if DEBUG:
                fig = self.v.view_graph(self.CG)
        except Exception as e:
            print(f"Creating graph failed \n {e} \n" )
            traceback.print_exc()
