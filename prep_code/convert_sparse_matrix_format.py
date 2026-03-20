import h5py
import scipy.sparse as sp
import numpy as np

def dump_tree(h5file):
    h5file.visititems(lambda name, obj: print(name))

input_h5 = "/lustre/home/BPM/autism_brain/nucleiQC_noFeatureQC/cas.h5"
output_h5 = "/lustre/home/ramachandruss/python_single_cell/test_interface/cas_csr.h5"

with h5py.File(input_h5, "r") as f:
    Xg = f["X"]

    data    = Xg["data"][:]
    indices = Xg["indices"][:]
    indptr  = Xg["indptr"][:]
    shape   = tuple(Xg["shape"][:])

# Reconstruct CSC (do NOT guess CSR here)
X_csc = sp.csc_matrix((data, indices, indptr), shape=shape)
X_csr = X_csc.tocsr()

with h5py.File(input_h5, "r") as fin, h5py.File(output_h5, "w") as fout:

    # ---- Copy everything except X ----
    for key in fin.keys():
        if key != "X":
            fin.copy(key, fout)

    # ---- Recreate X group ----
    Xgrp = fout.create_group("X")

    Xgrp.create_dataset("data",    data=X_csr.data)
    Xgrp.create_dataset("indices", data=X_csr.indices)
    Xgrp.create_dataset("indptr",  data=X_csr.indptr)
    Xgrp.create_dataset("shape",   data=np.array(X_csr.shape))

    # ---- Copy attributes on X if any ----
    for attr, val in fin["X"].attrs.items():
        Xgrp.attrs[attr] = val


with h5py.File(input_h5, "r") as f:
    print("INPUT:")
    dump_tree(f)

with h5py.File(output_h5, "r") as f:
    print("\nOUTPUT:")
    dump_tree(f)


