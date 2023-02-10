def get_labels(molecule):
    """Return the list of all labels of the nodes of a molecule (without specific order)."""
    return [node[1]["labels"][0] for node in molecule.nodes(data=True)]
