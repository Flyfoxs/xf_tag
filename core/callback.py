from keras.callbacks import Callback
import keras
from file_cache.utils.util_log import logger,timed
import pandas as pd

from core.feature import accuracy

class Cal_acc(Callback):

    def __init__(self, val_x, y, X_test):
        super(Cal_acc, self).__init__()
        self.val_x , self.y, self.X_test = val_x, y, X_test

        #logger.info(f'Cal_acc base on X:{self.X.shape}, Y:{self.y.shape}')

    @timed()
    def cal_acc(self):
        input1_col = [col for col in self.val_x.columns if not str(col).startswith('tfidf_')]
        input2_col = [col for col in self.val_x.columns if str(col).startswith('tfidf_')]
        model = self.model
        res = model.predict([self.val_x.loc[:,input1_col], self.val_x.loc[:,input2_col]])

        res = pd.DataFrame(res, index=self.val_x.index)
        acc1, acc2, total = accuracy(res, self.y)
        logger.info(f'acc1:{acc1:6.5f}, acc2:{acc2:6.5f}, <<<total:{total:6.5f}>>>')

        if total >=0.1:
            from core.attention import gen_sub
            gen_sub(model, self.X_test, f'{total:6.5f}', partition_len=1000*total )

        return acc1, acc2, total


    # def on_train_end(self, logs=None):
    #     acc1, acc2, total = self.cal_acc()
    #     return round(total, 5)

    def on_epoch_end(self, epoch, logs=None):
        acc1, acc2, total = self.cal_acc()
        return round(total, 5)

