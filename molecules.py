import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class Molecule(nx.classes.graph.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
    
    def nodelabels(self):
        nodeview = self.nodes(data=True)
        labels = [node[1]["labels"][0] for node in nodeview]
        return np.array(labels)
    
    def plot_graph(self):
        plt.figure()
        pos= nx.spring_layout(self)
        edge_to_labels = nx.get_edge_attributes(self,'labels')
        edge_labels = {e: edge_to_labels[e] for e in nx.get_edge_attributes(self,'labels')}
        RGB = ["r","g","b"]
        edge_colors = [RGB[e[2]['labels'][0]] for e in self.edges(data=True)]

        nx.draw(
            self, pos, width=1, linewidths=1,
            node_size=500, node_color='yellow', edge_color = edge_colors, alpha=0.8, font_size=15,
            labels={node[0]: node[1]['labels'][0] for node in self.nodes(data=True)},arrow_size=20
        )

        plt.axis('off')
        plt.show()



