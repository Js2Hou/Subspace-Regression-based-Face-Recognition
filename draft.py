import numpy as np

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge

from data_loader import DataLoader
from models import RRClassifier

train_x, train_y, test_x, test_y = DataLoader.load_AR(mode='1')
# rc = RidgeClassifier()
# rc.fit(train_x, train_y)
# pred_y = rc.predict(test_x)
# acc = 1 - np.count_nonzero(pred_y - test_y)/pred_y.shape[0]
# print(f'acc: {acc}')

T = RRClassifier.get_regular_simplex_vertices(120)
rc2 = Ridge()
train_y2 = T[train_y]
rc2.fit(train_x, train_y2)
pred_t = rc2.predict(test_x)
pred_y = []
for t_hat in pred_t:
    min_dist = 1e10
    pred_id = -1
    for id, t in enumerate(T):
        dist = np.linalg.norm(t - t_hat)
        if dist < min_dist:
            min_dist = dist
            pred_id = id
    pred_y.append(pred_id)
pred_y = np.array(pred_y)
acc = 1 - np.count_nonzero(pred_y - test_y) / pred_y.shape[0]
print(f'acc: {acc}')
