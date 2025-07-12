import numpy as np
from parse_helpers import parse_whole_hdf5
if __name__ == "__main__":
    data = parse_whole_hdf5("SM_subset.hdf5")
    np.savez("processed_SPICE2.npz", **data)

    #to get a pilot dataset
    key_list = [1,101,151,201,251,301,351,401,451]
    l = list(data.keys())
    sele = [l[i] for i in key_list]
    d = {}
    for k in sele: d[k] = data[k]
    np.savez("pilot1.npz",**d)