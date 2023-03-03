# /!\ WARNING /!\ 
# In this code, the term 'label' is used to designate what the NetworkX library calls node and edge 'attributes'.
# Thus, the node labels encode the atom types, and the edge labels encode the bond types.
# The 'primary key' used to identify nodes is called 'node index' (instead of 'node label' in the NetworkX library).
# The reason for these choices is to remain consistent with the vocabulary of the data challenge.
# ---------------

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
    
    def atom_dct(self):
        """Return self.atoms as a dictionary of the form {node_index:node_label, ...}."""
        return {a[0]:a[1] for a in self.atoms}
    
    def wl_relabeling(self, relabeling_dct=None, min_lbl=0):
        """
        Weisfeiler-Lehman relabeling.

        For each node, one compute a `neighborhood tuple` of the form:
            (node label, (edge label 1, neighbor label 1), (edge label 2, neighbor label 2), ...)
        Then, these neighborhood tuples are converted into new integer labels for the nodes.
        The edges remain unchanged.

        Parameters
        ----------
        relabeling_dct: None or dictionary
            dictionary whose keys are neighborhood tuples and whose values are the new labels that have to
            be assigned to them. This argument enables to have a consistent relabeling of two (or more) 
            different molecules.
        
        min_lbl: int
            minimum value of label to use for the relabeling

        Returns: instance of Molecule, dict
            a tuple of two elements:
                * a shallow copy of `self` with new node labels computed according to the Weisfeiler-Lehman relabeling.
                * the relabeling dictionary mapping neighborhood tuples to new labels.
        """
        if relabeling_dct is None:
            relabeling_dct = {}
        
        atom_dct = self.atom_dct()

        # compute dictionary with neighborhoods as lists
        nbh_lists = {a[0]:[] for a in self.atoms}  
        for bond in self.bonds:
            si, ti, bl = bond  # source index, target index, bond label
            sl = atom_dct[si]  # source label
            tl = atom_dct[ti]  # target label
            nbh_lists[si].append((bl, tl))
            nbh_lists[ti].append((bl, sl))
        
        # compute dictionary with neighborhoods as tuples (sorted to ensure unicity of the representation)
        nbh_tuples = {idx:(atom_dct[idx],) + tuple(sorted(nbh_lists[idx])) for idx in nbh_lists}
        
        # compute relabeling dictionary
        next_lbl = min_lbl
        for t in nbh_tuples.values():
            if not t in relabeling_dct:
                relabeling_dct[t] = next_lbl
                next_lbl += 1
        
        # compute new node labels
        new_node_labels = {idx:[relabeling_dct[nbh_tuples[idx]]] for idx in atom_dct}
        
        # create relabeld copy of Molecule
        relabeled_graph = self.copy()
        nx.set_node_attributes(relabeled_graph, new_node_labels, name='labels')
        return Molecule(relabeled_graph), relabeling_dct, next_lbl
    
    def node_label_frequencies(self):
        """Return dictionary of number of occurences of each node label.
        """
        res = {}
        for _, lbl in self.atoms:
            if not lbl in res:
                res[lbl] = 1
            else:
                res[lbl] += 1
        return res
    
    def plot(self, show_node_indices=False, **kwargs):
        """
        Plot molecule as graph.

        Parameters
        ----------
        show_node_indices: bool
            if True, the displayed node labels are tuples (node index, node label).
        
        **kwargs: dict
            keyword arguments to pass to nx.draw.
        """
        # Define keyword arguments to pass to function nx.draw
        default_kwargs = {"width":1, "linewidths":1, "font_size":15, "node_size":500}
        default_kwargs.update(kwargs)

        # Initialize figure
        plt.figure()

        # Compute node positions
        pos = nx.spring_layout(self)

        # Compute node labels, depending on whether to show node indices or not
        if show_node_indices:
            node_labels = {atom[0]: (atom[0], atom[1]) for atom in self.atoms}
        else:
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
