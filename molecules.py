import networkx as nx
import numpy as np

class Molecule(nx.classes.graph.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
    
    def nodelabels(self):
        nodeview = self.nodes(data=True)
        labels = [node[1]["labels"][0] for node in nodeview]
        return np.array(labels)

