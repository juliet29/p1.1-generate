import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import imageio.v3 as iio

import skimage.segmentation as seg
from skimage.segmentation import mark_boundaries
import skimage.measure as meas

RGB_CHANNEL_SIZE = 3

class Bbox():
    def __init__(self, arr):
        assert len(arr) == 4
        self.min_row=arr[0]
        self.min_col=arr[1]
        self.max_row=arr[2]
        self.max_col=arr[3]

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
        self.region_props = meas.regionprops(self.tensor_labels)


        pass

    def shapely2graph(self):
        pass


    # VISUALIZATIONS

    def view_plan_image(self):
        assert self.tensor_labels
        border_color = [0,1,1]
        plt.imshow(mark_boundaries(self.tensor, self.tensor_labels,color=border_color))
        return plt 


    def view_plan_segments(self):
        pass

    def view_plan_shapely(self):
        pass

    def view_graph(self):
        pass

