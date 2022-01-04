import numpy as np
from sklearn.model_selection import KFold

# X = np.load('/content/DAT/se2.npy')
import numpy as np
from sklearn.model_selection import KFold
X = np.load('/content/DAT/mnist_se2.npy')
y = np.load('/content/DAT/mnist_se2_init.npy')
kf = KFold(n_splits=4)
kf.get_n_splits(X)

print(kf)
kn = 1
for train_index, test_index in kf.split(X):  
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    np.save('/content/Saver/se2_fold'+str(kn)+'.npy',X_train)
    np.save('/content/Saver/se2_valid_fold'+str(kn)+'.npy',X_test)
    np.save('/content/Saver/se2_init_fold'+str(kn)+'.npy',y_train)
    np.save('/content/Saver/se2_init_valid_fold'+str(kn)+'.npy',y_test)
    kn=kn+1