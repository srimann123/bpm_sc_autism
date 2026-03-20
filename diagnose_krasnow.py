import h5py

input_h5 = "/lustre/home/ramachandruss/python_single_cell/krasnow_transposed.h5ad"
with h5py.File(input_h5, "r") as f:
    if 'X' in f and 'encoding-type' in f['X'].attrs:
        encoding = f['X'].attrs['encoding-type']
        # Decode bytes if necessary
        if isinstance(encoding, bytes):
            encoding = encoding.decode('utf-8')
        print(encoding)
    else:
        print("Not a standard sparse H5AD or dense")
