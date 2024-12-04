from sklearn.cross_decomposition import PLSRegression
import numpy as np


# Calculate VIP scores
def calculate_vip(pls, X):
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([w[i, j] ** 2 * s[j] for j in range(h)])
        vip[i] = np.sqrt(p * np.sum(weight) / total_s)
    return vip


# VIP score
X_train = "inputdata"
y_train = "Label"
# 执行PLSR
pls = PLSRegression()
pls.fit(X_train, y_train)
vip_scores = calculate_vip(pls, X_train)