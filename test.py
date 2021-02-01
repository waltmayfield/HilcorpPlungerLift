
# import PrepTFRecord as PR
# import pandas as pd; import numpy as np

# fname = r"C:\Users\wmayfield\Downloads\3003921558(2).csv"

# print(pd.read_csv(fname).iloc[:,:10].head())

# print(f'csv shape: {pd.read_csv(fname).iloc[:,4:].shape}')

# # X = PR.csv_to_tensor(fname)
# # print(f'X shape: {X.shape}')
# # print(X[:,:5])
# # print(f'All values finite?:{np.isfinite(X).min()==1} ')

# X, Y, path = PR.process_path(fname)
# if min(np.isfinite(X).min(),np.isfinite(Y).min()) == 0:
# 	print(f'Found NaN or Inf with file : {path}')	
# else:
# 	print('No Nan or Inf')

# print(f'X error location: {np.isfinite(X).argmax()}')

# print(X[0,:])
# print(X.shape)
# print(Y.shape)
# print(path)
# print("Done")


print((None or 1))

# import tensorflow as tf

# X = tf.expand_dims(tf.constant([1,2,3,4]), axis = 0)

# print(tf.repeat(X, repeats = [10], axis = 0))


# import numpy as np
# from sklearn.impute import KNNImputer
# X = [[1, 2, np.nan], [3, 4, np.nan], [np.nan, 6, np.nan], [8, 8, np.nan]]
# print(X)
# imputer = KNNImputer(n_neighbors=2)
# print(imputer.fit_transform(X))
