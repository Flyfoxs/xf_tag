from keras.callbacks import Callback
import keras
from file_cache.utils.util_log import logger,timed
import pandas as pd

from core.feature import accuracy

class Cal_acc(Callback):

    def __init__(self, X, y):
        super(Cal_acc, self).__init__()
        self.X , self.y = X, y
        logger.info(f'Cal_acc base on X:{self.X.shape}, Y:{self.y.shape}')

    @timed()
    def cal_acc(self):
        model: keras.Model = self.model
        res = model.predict(self.X)

        res = pd.DataFrame(res, index=self.X.index)
        acc1, acc2, total = accuracy(res, self.y)
        logger.info(f'acc1:{acc1:6.5f}, acc2:{acc2:6.5f}, total:{total:6.5f}')
        return acc1, acc2, total


    # def on_train_end(self, logs=None):
    #     acc1, acc2, total = self.cal_acc()
    #     return round(total, 5)

    def on_epoch_end(self, epoch, logs=None):
        acc1, acc2, total = self.cal_acc()
        return round(total, 5)

