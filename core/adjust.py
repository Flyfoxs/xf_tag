
from pandas import DataFrame as DF
from core.conf import num_classes

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
        self.coef_arr = []
        self.best_score = 0
        self.initial_score = 0
        self.val_score = []
        self.initial_coef = [1] * num_classes

    def _kappa_loss(self, coef, X, y):
        X_p = DF(np.copy(X))
        for i in range(len(coef)):
            X_p[i] *= coef[i]

        l1 = f1_score(y, np.argmax(X_p.values, axis=1), average="weighted")
        self.coef_arr.append(coef)
        self.best_score = max(l1, self.best_score)
        print(list(coef.astype(np.float16)), ' Train score = ', l1.astype(np.float32), 'Best Score',
              self.best_score)  # ,' Valid score =',l2.astype(np.float16))
        return -l1

    def fit(self, X, y):
        self.initial_score = f1_score(y, np.argmax(X.values, axis=1), average="weighted")

        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = sp.optimize.minimize(loss_partial, self.initial_coef, method='Powell')

    def predict(self, X, coef):

        logger.info(f'Predict with:{coef}')
        #print(self.coef_)
        X_p = DF(np.copy(X))
        for i in range(len(coef)):
            X_p[i] *= coef[i]
        return X_p

    def coefficients(self):
        return self.coef_['x']