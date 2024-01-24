import re
from enum import Enum

from shapely import Polygon
import numpy as np
import plotly.express as px
import networkx as nx


RGB_CHANNEL_SIZE = 3

DOOR_CODE = 177 # repeated 3 times to make the RGB for doors

COLORWAY = ['#702632', '#A4B494', '#495867',  "#81909E", "#F4442E", "#DB7C26", "#BB9BB0"]

GM = (np.sqrt(5)-1.0)/2.0
W = 8
H = W*GM
SIZE = (W, H)

class Bbox():
    def __init__(self, arr):
        assert len(arr) == 4
        self.min_row=arr[0]
        self.min_col=arr[1]
        self.max_row=arr[2]
        self.max_col=arr[3]

class RegionType(Enum):
    ROOM = 0
    DOOR = 1

class Region():
    bbox:list = None
    coords:list = None
    shape:Polygon = None
    centroid:tuple = None
    unique_colors:set = None
    type:RegionType = None
    label:str = None
    area:float = None

class PlanData:
    tensor:np.ndarray = None
    tensor_labels:np.ndarray = None
    tensor_labels_w_doors:np.ndarray = None
    regions:[Region] = None
    graph:nx.Graph = None


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


def check_adjacency(curr_region:Region.shape, nb_region:Region.shape):
    # shapely check for adjacency 
    # check = curr_region.touches(nb_region)
    check = curr_region.intersects(nb_region)
    return int(check) # True = 1, False = 0

def rgb_str_to_rgba(rgb_str):
    # Extract numeric values from the string using regular expression
    rgb_values = [int(value) for value in re.findall(r'\d+', rgb_str)]
    
    # Convert to RGBA format
    rgba_values = [value / 255 for value in rgb_values] + [1.0]  # Normalize RGB values to the range [0, 1]
    
    return tuple(rgba_values)

def create_colorway(n_colors=20):
    colors = px.colors.sample_colorscale("turbo", [n/(n_colors -1) for n in range(n_colors)])
    return [rgb_str_to_rgba(r) for r in colors]



def print_keys_values(obj):
    if isinstance(obj, dict):
        # If it's a dictionary
        for key, value in obj.items():
            print(f"{key}: {value}")
    elif hasattr(obj, '__dict__'):
        # If it's an instance of a class
        for key, value in vars(obj).items():
            print(f"{key}: {value}")
    else:
        print("Unsupported object type")

def print_attribute_from_instances(instances, attribute_name):
    for instance in instances:
        # Use getattr to access the attribute dynamically
        attribute_value = getattr(instance, attribute_name, None)
        if attribute_value is not None:
            print(f"{attribute_name} for {instance.__class__.__name__}: {attribute_value}")
        else:
            print(f"{attribute_name} not found for {instance.__class__.__name__}")

def print_many_attributes_from_instances(instances, attribute_names):
    # print(f"\nAttributes for {instance.__class__.__name__}:")
    for ix, instance in enumerate(instances):
        for attribute_name in attribute_names:
            # Use getattr to access the attribute dynamically
            attribute_value = getattr(instance, attribute_name, None)
            if attribute_value is not None:
                print(f"{ix}: {attribute_name}: {attribute_value}")
            else:
                print(f"{ix}: {attribute_name} not found")