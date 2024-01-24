import networkx as nx

from helpers import *

class GraphLogic:
    def __init__(self, regions:[Region]):
        self.regions= regions

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
        return self.CG

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