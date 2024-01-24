import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import shapely.plotting as splt

from skimage.segmentation import mark_boundaries

import networkx as nx

from helpers import *



class Visualizer:
    def __init__(self):
        pass
        # plan_data = plan_data

    def view_plan_image(self,  tensor:np.ndarray):
        plt.imshow(tensor)
        return plt 
    

    def view_plan_segments(self, tensor:np.ndarray, tensor_labels_w_doors:np.ndarray ):
        label_nums = np.unique(tensor_labels_w_doors)
        
        norm = LogNorm(vmin=np.min(label_nums), vmax=np.max(label_nums))
        cmap = plt.get_cmap("turbo")

        fig, ax = plt.subplots()
        im = ax.imshow(tensor_labels_w_doors, cmap=cmap, norm=norm)
        ax.figure.colorbar(im)

        return fig

    def view_plan_shapely(self, regions:[Region]):
        assert type(regions) != None

        fig = plt.figure(figsize=SIZE,  dpi=90)
        ax = fig.add_subplot(111)
        colors = create_colorway(len(regions))

        for ix, region in enumerate(regions):
            splt.plot_polygon(region.shape, ax=ax, alpha=0.5, color=colors[ix])
            ax.annotate(str(region.label), region.centroid)

        return fig 

    def view_graph(self, graph:nx.Graph):
        fig = plt.figure(figsize=SIZE,  dpi=90)

        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos, )
        nx.draw_networkx_labels(graph, pos)
        return plt
    