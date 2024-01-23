import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import imageio.v3 as iio

from matplotlib.colors import LogNorm

from sklearn.cluster import KMeans

import skimage.segmentation as seg
from skimage.segmentation import mark_boundaries
import skimage.measure as meas

from shapely import Polygon
import shapely.plotting as splt

import networkx as nx

from helpers import *




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
        
        self.df = pd.DataFrame(self.tensor2D, columns=["R", "G", "B"])
        self.df["Label"] = np.zeros(len(self.df), dtype=np.int8)

        groups = self.df.groupby(["R", "G", "B"]).groups
        for ix, group_name in enumerate(groups.keys()):
            for i in list(groups[group_name]):
                self.df.at[i, "Label"] = int(ix+1)

        self.tensor_labels = np.array(self.df["Label"]).reshape(self.tensor_shape[:2])
        return 
    
    def further_segment_doors(self, num_doors=6):
        target_values = np.array([DOOR_CODE]*3)
        check = np.all(self.tensor == target_values, axis=-1)
        self.indices_split = np.where(check)
        self.indices = np.argwhere(check)

        # cluster and re-label doors so they have their own regions individual 
        np.random.seed(42)
        kmeans = KMeans(n_clusters=num_doors, n_init=10, init="k-means++") # TODO figure out why 
        y_pred = kmeans.fit_predict(self.indices)

        self.tensor_labels_w_doors = self.tensor_labels.copy()
        self.tensor_labels_w_doors = self.tensor_labels_w_doors.astype(np.int64)

        for x, y, door_label in zip(self.indices_split[0], self.indices_split[1], y_pred):
            self.tensor_labels_w_doors[x,y] = DOOR_CODE+door_label 



    def array2shapely(self):
        assert type(self.tensor_labels) != None
        self.all_region_props = meas.regionprops(self.tensor_labels_w_doors, self.tensor)

        self.regions = []
        for ix, region in enumerate(self.all_region_props): 
            r = Region()

            # keep track of the colors in each region 
            int0 = region.image_intensity
            s = int0.shape
            int01 = np.reshape(int0, (s[0] * s[1], 3))
            unique_tuples = {tuple(arr) for arr in int01}
            r.unique_colors = unique_tuples

            r.bbox = region.bbox
            r.coords = create_coords(Bbox(r.bbox))
            r.shape = Polygon(r.coords)
            r.centroid = (r.shape.centroid.x, r.shape.centroid.y)
            r.label = region.label
            r.area = region.bbox_area

            # TODO code this as doors
            if r.unique_colors == {(177, 177, 177)}:
                r.type = RegionType.DOOR
            else:
                r.type = RegionType.ROOM

            # ignore all regions that are not real rooms 
            if r.unique_colors != {(0,0,0)}:
                self.regions.append(r)
        return 

            
    def shapely2graph(self):
        self.init_graph()
        g1 = self.view_graph()
        self.check_graph_adj()
        g2 = self.view_graph()
        self.filter_graph()
        g3 = self.view_graph()
        return g1, g2, g3



    # GRAPH MANIPULATION 
    def init_graph(self):
        # initialize fully connected graph with edge adjacency = 0
        n_nodes = len(self.regions) 
        self.CG = nx.complete_graph(n_nodes)  

        region_data = [{"type": getattr(instance, "type", None), "label": getattr(instance, "label", None)} for instance in self.regions]

        node_attrs = dict(zip(list(self.CG.nodes), region_data))
        nx.set_node_attributes(self.CG, node_attrs)

        # need to have labels / indices for the edge properties that explicitly index into the matrix 
        n = np.linspace(0, n_nodes-1, n_nodes)
        X2D,Y2D = np.meshgrid(n,n)
        attrs = {(x,y): {"adj": 0} for x,y,in 
                zip(Y2D.ravel(), X2D.ravel())}

        nx.set_edge_attributes(self.CG, attrs)
        return 

    def check_graph_adj(self):
        for n in self.CG.nodes:
            neighbors = list(nx.bfs_edges(self.CG, source=0, depth_limit=1))
            for pair in neighbors:
                if self.CG.edges[pair]["adj"] == 0:
                    adj_val = check_adjacency(
                        self.regions[pair[0]].shape,
                        self.regions[pair[1]].shape)
                    
                    self.CG.edges[pair]["adj"] = adj_val
        return 

    def filter_graph(self):
        for e in self.CG.edges:
            if self.CG.edges[e]["adj"] == 0:
                self.CG.remove_edge(*e)

        pass


    # VISUALIZATIONS
    def view_plan_image(self):
        assert type(self.tensor) != None
        plt.imshow(self.tensor)
        return plt 


    def view_plan_segments(self):
        assert type(self.tensor_labels) != None

        border_color = [0,1,1]
        label_nums = np.unique(self.tensor_labels_w_doors)
        
        norm = LogNorm(vmin=np.min(label_nums), vmax=np.max(label_nums))
        cmap = plt.get_cmap("turbo")

        fig, ax = plt.subplots()
        im = ax.imshow(mark_boundaries(self.tensor, self.tensor_labels,color=border_color))

        fig2, ax = plt.subplots()
        im = ax.imshow(self.tensor_labels_w_doors, cmap=cmap, norm=norm)
        cbar = ax.figure.colorbar(im)

        return fig, fig2 

    def view_plan_shapely(self):
        assert type(self.regions) != None

        fig = plt.figure(figsize=SIZE,  dpi=90)
        ax = fig.add_subplot(111)
        colors = create_colorway(len(self.regions))

        for ix, region in enumerate(self.regions):
            splt.plot_polygon(region.shape, ax=ax, alpha=0.5, color=colors[ix])
            ax.annotate(str(region.label), region.centroid)

        return fig 

    def view_graph(self):
        assert self.CG
        fig = plt.figure(figsize=SIZE,  dpi=90)

        pos = nx.spring_layout(self.CG)
        nx.draw_networkx_nodes(self.CG, pos)
        nx.draw_networkx_edges(self.CG, pos, )
        nx.draw_networkx_labels(self.CG, pos)
        self.CG_edge_labels = nx.draw_networkx_edge_labels(self.CG, pos)
        return plt
    

