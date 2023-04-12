def histogram_kernel(mol1, mol2):
    freq1 = mol1.node_label_frequencies()
    freq2 = mol2.node_label_frequencies()
    res = 0
    for lbl in freq1:
        if lbl in freq2:
            res += freq1[lbl] * freq2[lbl]
    return res

def wlsk(mol1, mol2, n_iter=3):
    """Weisfeiler Lehman subtree kernel"""
    res = histogram_kernel(mol1, mol2)
    for _ in range(n_iter):
        mol1, relabeling_dct, next_lbl = mol1.wl_relabeling()
        mol2, _, _ = mol2.wl_relabeling(relabeling_dct, next_lbl)
        res += histogram_kernel(mol1, mol2)

    return res
