import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


NODECM = cm.get_cmap('Set1')  # colormap for nodes
EDGECM = ['red', 'green', 'blue', 'yellow']  # colormap for edges


class Molecule(nx.classes.graph.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.atoms = [(node[0], node[1]['labels'][0]) for node in self.nodes(data=True)]
        self.bonds = [(edge[0], edge[1], edge[2]['labels'][0]) for edge in self.edges(data=True)]
    
    def plot(self, **kwargs):
        """
        Plot molecule as graph.
        """
        # Define keyword arguments to pass to function nx.draw
        default_kwargs = {"width":1, "linewidths":1, "font_size":15, "node_size":500}
        default_kwargs.update(kwargs)

        # Initialize figure
        plt.figure()

        # Compute node positions
        pos = nx.spring_layout(self)

        # Compute node labels
        node_labels = {atom[0]: atom[1] for atom in self.atoms}

        # Compute node and edge colors
        node_colors = [NODECM(atom[1]) for atom in self.atoms]
        edge_colors = [EDGECM[bond[2]] for bond in self.bonds]

        # Draw molecule
        nx.draw(
            self, pos, node_color=node_colors, edge_color=edge_colors, labels=node_labels, **default_kwargs
        )

        # Show plot
        plt.axis('off')
        plt.show()
