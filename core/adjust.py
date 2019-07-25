
from pandas import DataFrame as DF


from core.conf import num_classes
from core.ensemble import *

adjust_col = [
'140901',
'142302',
'142701',
'140601',
'140206',
'142601',
'140210',
'140211',
'142105',
'140207'
]
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
        self.coef_arr = []
        self.best_score = 0
        self.initial_score = 0
        self.val_score = []
        self.initial_coef = [.88530247, 1.09971672, 0.91296856, 1.42351244, 1.30119761,
                             1.67477818,1.13638098, 1.38549708, 1.02128626, 1.47433436]

    #@timed()
    def _kappa_loss(self, coef, X):
        X = X.copy()

        for col, adjust in zip(adjust_col, coef):
            X.loc[:, col] = X.loc[:, col] * adjust

        acc1, acc2, total = accuracy(X)
        logger.info(f'acc1:{acc1:7.6f}, acc2:{acc2:7.6f}, total:{total:7.6f} with:{coef}')
        return -round(total,6)

    def fit(self, X):
        self.initial_score = accuracy(X)
        logger.info(f'initial_score:{self.initial_score}')
        from functools import partial
        loss_partial = partial(self._kappa_loss, X=X)
        import scipy
        self.coef_ = scipy.optimize.minimize(loss_partial, self.initial_coef, method='Powell')

    def predict(self, test, coef):

        logger.info(f'Predict with:{coef}')
        test = test.copy()
        for col, adjust in zip(adjust_col, coef):
            test.loc[:, col] = test.loc[:, col] * adjust
        return test

    def coefficients(self):
        return self.coef_['x']



if __name__ == '__main__':
    top=2
    opt = OptimizedRounder()
    oof = get_feature_oof(top)
    train = oof.loc[oof.label != '0']
    opt.fit(oof)
    coef = opt.coefficients()
    logger.info(f'Best coef:{coef}')

    test = oof.loc[oof.label == '0']
    for col, adjust in zip(adjust_col, coef):
        test.loc[:, col] = test.loc[:, col] * adjust

    gen_sub_file(test, f'mean_adjust_{top}.csv')





""""
nohup python -u ./core/adjust.py > adjust.log 2>&1 &
"""