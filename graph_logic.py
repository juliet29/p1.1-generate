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
    
    def study_graph_adj(self):
        self.door_adj_nodes = []
        self.door_adj_labels = []

        for node_ix, attrs in self.CG.nodes(data=True):
            if attrs["type"] == RegionType.DOOR:
                neighbors = list(nx.bfs_edges(self.CG, source=node_ix, depth_limit=1))

                for pair in neighbors:
                    if self.CG.edges[pair]["adj"] == 0:
                        ni = pair[0]
                        nj = pair[1]
                        adj_val = check_adjacency(
                            self.regions[ni].shape,
                            self.regions[nj].shape)
                        if adj_val:
                            self.door_adj_labels.append((self.CG.nodes[ni]['label'], self.CG.nodes[nj]['label']))
                            self.door_adj_nodes.append((ni, nj))
                        
                        self.CG.edges[pair]["adj"] = adj_val

        return self.door_adj_labels


    def create_correct_adj(self):
        grouped_tuples = {}
        for tup in self.door_adj_nodes:
            first_item = tup[0]
            if first_item not in grouped_tuples:
                grouped_tuples[first_item] = [tup[1]]
            else:
                grouped_tuples[first_item].append(tup[1])

        prelim_edges = {tuple(i): {"adj": 1} for i in grouped_tuples.values()}
        self.correct_edges  = {key: value for key, value in prelim_edges.items() if len(key) > 1}

        return self.correct_edges


    def create_correct_graph(self):
        self.CG.remove_edges_from(self.CG.edges())
        self.CG.add_edges_from(self.correct_edges)

        return self.CG