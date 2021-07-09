using MAT: matread

data_file = "./data/musk.mat"
vars = matread(data_file)

X, y = copy(vars["X"]'), vars["y"]
X, y


