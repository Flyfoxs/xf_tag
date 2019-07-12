from keras.callbacks import Callback
import keras
from file_cache.utils.util_log import logger,timed
import pandas as pd

from core.feature import accuracy

class Cal_acc(Callback):

    def __init__(self, val_x, y, X_test):
        super(Cal_acc, self).__init__()
        self.val_x , self.y, self.X_test = val_x, y, X_test

        self.feature_len = self.val_x.shape[1]

        import time, os
        self.batch_id = round(time.time())
        self.model_folder = f'./output/model/{self.batch_id}/'

        os.makedirs(self.model_folder)


        #logger.info(f'Cal_acc base on X:{self.X.shape}, Y:{self.y.shape}')

    @timed()
    def cal_acc(self):
        input1_col = [col for col in self.val_x.columns if not str(col).startswith('fea_')]
        input2_col = [col for col in self.val_x.columns if str(col).startswith('fea_')]
        model = self.model
        res = model.predict([self.val_x.loc[:,input1_col], self.val_x.loc[:,input2_col]])

        res = pd.DataFrame(res, index=self.val_x.index)
        acc1, acc2, total = accuracy(res, self.y)
       # logger.info(f'acc1:{acc1:6.5f}, acc2:{acc2:6.5f}, <<<total:{total:6.5f}>>>')

        return acc1, acc2, total


    # def on_train_end(self, logs=None):
    #     acc1, acc2, total = self.cal_acc()
    #     return round(total, 5)

    def on_epoch_end(self, epoch, logs=None):
        acc1, acc2, total = self.cal_acc()
        logger.info(f'Epoch#{epoch}, acc1:{acc1:6.5f}, acc2:{acc2:6.5f}, <<<total:{total:6.5f}>>>')

        if total >= 0.65:
            model_path = f'{self.model_folder}/model_{self.feature_len}_{total:6.5f}_{epoch}.h5'
            self.model.save(model_path)
            print(f'weight save to {model_path}')

        threshold = 0.7
        if total >=threshold:
            logger.info(f'Try to gen sub file for local score:{total}, and save to:{model_path}')
            from core.attention import gen_sub
            gen_sub(model_path)
        else:
            logger.info(f'Only gen sub file if the local score >={threshold}, current score:{total}')


        return round(total, 5)

