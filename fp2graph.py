import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import imageio.v3 as iio

import skimage.segmentation as seg
from skimage.segmentation import mark_boundaries
import skimage.measure as meas

from shapely import Polygon
import shapely.plotting as splt

import networkx as nx

RGB_CHANNEL_SIZE = 3
COLORWAY = ['#702632', '#A4B494', '#495867', '#912F40', "#81909E", "#F4442E", "#DB7C26", "#BB9BB0"]

class Bbox():
    def __init__(self, arr):
        assert len(arr) == 4
        self.min_row=arr[0]
        self.min_col=arr[1]
        self.max_row=arr[2]
        self.max_col=arr[3]




class Region():
    def __init__(self):
        self.bbox:list = None
        self.coords:list = None
        self.shape:Polygon = None
        self.centroid:tuple = None



def create_coords(b:Bbox): 
    # clockwise from origin in bottom-left corner to match example: 
    # https://shapely.readthedocs.io/en/stable/reference/shapely.LinearRing.html#shapely.LinearRing
    coords = [
        # col ==> x, row ==> y 
        (b.min_col, b.min_row),
        (b.max_col, b.min_row),
        (b.max_col, b.max_row),
        (b.min_col, b.max_row),
    ]
    return coords

# def check_adjacency(curr_node_ix:int, nb_node_ix:int):
#     # shapely check for adjacency 
#     check = reg_d[curr_node_ix]["shape"].touches(reg_d[nb_node_ix]["shape"])
#     return int(check) # True = 1, False = 0

def check_adjacency(curr_region:Region.shape, nb_region:Region.shape):
    # shapely check for adjacency 
    check = curr_region.touches(nb_region)
    return int(check) # True = 1, False = 0




class FloorPlan2Graph:
    def __init__(self, PATH):
        self.PATH = PATH

    # CONVERSION FUNCTIONS      
    def image2tensor(self):
        self.tensor = iio.imread(self.PATH, pilmode='RGB') 
        self.tensor_shape = self.tensor.shape
        assert self.tensor_shape[-1] == RGB_CHANNEL_SIZE
        return 

    def segment_tensor(self):
        self.tensor2D = self.tensor.reshape((-1, RGB_CHANNEL_SIZE))
        assert np.array_equal(
            self.tensor, 
            self.tensor2D.reshape(self.tensor.shape)
            )
        
        df = pd.DataFrame(self.tensor2D, columns=["R", "G", "B"])
        df["Label"] = np.zeros(len(df), dtype=np.int8)

        groups = df.groupby(["R", "G", "B"]).groups
        for ix, group_name in enumerate(groups.keys()):
            for i in list(groups[group_name]):
                df.at[i, "Label"] = int(ix)

        self.tensor_labels = np.array(df["Label"]).reshape(self.tensor_shape[:2])
        return 


    def array2shapely(self):
        assert self.tensor_labels
        # all_region_props called p in notebook 
        self.all_region_props = meas.regionprops(self.tensor_labels)

        self.regions = []
        for ix, region in enumerate(self.all_region_props): 
            # self.regions[ix] = Region()
             # self.regions[ix]
            r = Region()
            r.bbox = region.bbox
            r.coords = create_coords(Bbox(r.bbox))
            r.shape = Polygon(r.coords)
            r.centroid = (r.shape.centroid.x, r.shape.centroid.y)
            self.regions.append(r)
        return 

            
    def shapely2graph(self):
        self.init_graph()
        self.check_graph_adj()
        self.filter_graph()
        return



    # GRAPH MANIPULATION 
    def init_graph(self):
        # initialize fully connected graph with edge adjacency = 0
        n_nodes = len(self.regions) 
        self.CG = nx.complete_graph(n_nodes)  

        # set edge properties -> 0 = all unconnected to begin 
        init_adj = np.zeros((n_nodes, n_nodes))

        # need to have labels / indices for the edge properties that explicitly index into the matrix 
        n = np.linspace(0, n_nodes-1, n_nodes)
        X2D,Y2D = np.meshgrid(n,n)
        attrs = {(x,y): {"adj": i} for x,y,i in 
                zip(Y2D.ravel(), X2D.ravel(), init_adj.ravel())}

        nx.set_edge_attributes(self.CG, attrs)
        return 

    def check_graph_adj(self):
        for n in self.CG.nodes:
            neighbors = list(nx.bfs_edges(self.CG, source=0, depth_limit=1))
            for pair in neighbors:
                if self.CG.edges[pair]["adj"] == 0:
                    adj_val = check_adjacency(
                        self.regions[pair[0].shape],
                        self.regions[pair[1].shape])
                    
                    self.CG.edges[pair]["adj"] = adj_val
        return 

    def filter_graph(self):
        for e in self.CG.edges:
            if self.CG.edges[e]["adj"] == 0:
                self.CG.remove_edge(*e)

        pass


    # VISUALIZATIONS

    def view_plan_image(self):
        assert self.tensor
        plt.imshow(self.tensor)
        return plt 


    def view_plan_segments(self):
        assert self.tensor_labels
        border_color = [0,1,1]
        plt.imshow(mark_boundaries(self.tensor, self.tensor_labels,color=border_color))
        return plt 

    def view_plan_shapely(self):
        assert self.regions

        GM = (np.sqrt(5)-1.0)/2.0
        W = 8
        H = W*GM
        SIZE = (W, H)

        fig = plt.figure(figsize=SIZE,  dpi=90)
        ax = fig.add_subplot(111)

        for ix, region in enumerate(self.regions):
            splt.plot_polygon(region.shape, ax=ax, alpha=0.5, color=COLORWAY[ix])
            ax.annotate(str(ix), region.centroid)

        return fig 

    def view_graph(self):
        assert self.CG
        pos = nx.spring_layout(self.CG)
        nx.draw_networkx_nodes(self.CG, pos)
        nx.draw_networkx_edges(self.CG, pos, )
        nx.draw_networkx_labels(self.CG, pos)
        self.CG_edge_labels = nx.draw_networkx_edge_labels(self.CG, pos)
        return plt

