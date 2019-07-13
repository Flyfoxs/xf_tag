import sys
import os

from sklearn.model_selection import StratifiedKFold

from core.callback import *

from core.feature import *
from core.conf import *
import keras
import os



os.environ['TF_KERAS'] = '1'






@timed()
def get_train_test_bert(frac=1):

    raw_data = get_raw_data()

    bert = get_feature_bert(SEQ_LEN)

    manual = get_feature_manual()

    data = pd.concat([raw_data, bert, manual ], axis=1)

    logger.info(f'Shape of data:{data.shape}, raw_data:{raw_data.shape} manual:{manual.shape}, bert:{bert.shape} ')

    data.sort_values(['type_cnt'])



    train_data = data.loc[pd.notna(data.type_id)].sample(frac=frac, random_state=2019)
    labels = train_data.type_id.values.tolist()

    test_data =  data.loc[pd.isna(data.type_id)].sample(frac=frac, random_state=2019)

    logger.info(f'Train:{train_data.shape} Test:{test_data.shape}, frac:{frac}')

    feature_col = [col for col in data.columns if col.startswith('fea_') or col.startswith('bert_')]

    label2id, id2label = get_label_id()
    #word2id = get_word2id()

    # Encode input words and labels
    X = train_data.loc[:, feature_col]
    Y = [label2id[label] for label in labels]


    X_test = test_data.loc[:, feature_col]


    return  X, pd.Series(Y, index=train_data.index), X_test


# X, y, X_test = get_train_test_bert(0.1)
#
#
# train_x, train_y = load_data(train_path)
# test_x, test_y = load_data(test_path)



from keras_bert import load_trained_model_from_checkpoint

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True,seq_len=SEQ_LEN,)
model.summary(line_length=120)

from tensorflow.python import keras
from keras_bert import AdamWarmup, calc_train_steps
inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=152, activation='softmax')(dense)

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-4




def train_base(frac=1):
    X, y, X_test = get_train_test_bert(frac)

    decay_steps, warmup_steps = calc_train_steps(
        y.shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    model = keras.models.Model(inputs, outputs)
    model.compile(
        AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', 'accuracy'],
    )

    input1_col = [col for col in X.columns if str(col).startswith('bert_')]
    input2_col = [col for col in X.columns if str(col).startswith('fea_')]
    #max_words = len(input1_col)
    model #= get_model(max_words)

    #get_feature_manual.cache_clear()
    Y_cat = keras.utils.to_categorical(y, num_classes=num_classes)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
    for train_idx, test_idx  in  folds.split(X.values, y):

        logger.info(f'Shape train_x.loc[:, input1_col].iloc[:,0]: {X.loc[:, input1_col].iloc[:,0].shape}')
        train_x, train_y, val_x, val_y = \
            X.iloc[train_idx], Y_cat[train_idx], X.iloc[test_idx], Y_cat[test_idx]

        logger.info(f'get_train_test output: train_x:{train_x.shape}, train_y:{train_y.shape}, val_x:{val_x.shape}, X_test:{X_test.shape}')
        for sn in range(5):
            input1 = train_x.loc[:, input1_col]
            input2 = np.zeros_like(input1)

            logger.info(f'NN Input1:{input1.shape}, Input2:{input2.shape}')

            logger.info(f'NN Input1:{input1[:3]}')
            his = model.fit([input1, input2], train_y,
                            validation_data = ([val_x.loc[:, input1_col], np.zeros_like(val_x.loc[:, input1_col])], val_y),
                            epochs=8,  shuffle=True, batch_size=64,
                            #callbacks=[Cal_acc(val_x, y.iloc[test_idx], X_test)]
                      #steps_per_epoch=1000, validation_steps=10
                      )



            #gen_sub(model, X_test, sn)

            break


if __name__ == '__main__':
    train_base(1)

"""
nohup python -u ./core/bert.py  > bert.log 2>&1 &
"""