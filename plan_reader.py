import numpy as np
import pandas as pd

import imageio.v3 as iio

from sklearn.cluster import KMeans

import skimage.measure as meas

from helpers import *



class PlanReader:
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
        assert np.array_equal(self.tensor, 
            self.tensor2D.reshape(self.tensor.shape))
        
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


            if r.unique_colors == {(DOOR_CODE, DOOR_CODE, DOOR_CODE)}:
                r.type = RegionType.DOOR
            else:
                r.type = RegionType.ROOM

            # ignore all regions that are not real rooms ie -> dont have color assigned 
            if r.unique_colors != {(0,0,0)}:
                self.regions.append(r)


        return 
