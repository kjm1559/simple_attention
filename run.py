from source.utils import make_train_data, TIME_STEPS, decoding_string, encoding_embeding, string_index, make_test_data, decoding_embeding, levenshtein
from source.models import attention_model
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        a, b = make_test_data()
        model = attention_model('test')
        model.load_weights('attention_test.h5')
        index = np.random.randint(len(a))
        test_data = encoding_embeding('1854-05-08, 0', string_index)
        test = model.predict_on_batch(np.array([test_data]))
        answer = 'monday, 08 may 1854'
        print('input :', decoding_embeding(test_data, string_index))
        print('prdict :', decoding_string(test[0][0], string_index))
        print('answer :', answer)#decoding_string(answer, string_index))
        print(test[1][0].shape, np.sum(test[1][0], axis=0), np.sum(test[1][0], axis=1))
        test = model.predict_on_batch(np.array([a[index]]))

        y_true = decoding_embeding(a[index], string_index)
        y_predict = decoding_string(test[0][0], string_index)

        plt.imshow(test[1][0][:, ::-1][:len(y_true), :len(y_predict)])
        plt.yticks(np.arange(len(y_true)), y_true)
        plt.xticks(np.arange(len(y_predict)), y_predict)
        plt.savefig('attention_matrix.png')
    
    if sys.argv[1] == 'compare':
        a, b = make_test_data()
        model_att = attention_model('compare')
        model_gru = attention_model('compare', False)
        
        model_att.load_weights('attention_test.h5')
        model_gru.load_weights('gru_test.h5')
        
        att_r = model_att.predict(np.array(a), batch_size=32)
        gru_r = model_gru.predict(np.array(a), batch_size=32)
        
        att_d = 0
        gru_d = 0
        for i in range(len(b)):
            att_d += levenshtein(decoding_string(b[i], string_index), decoding_string(att_r[i], string_index))
            gru_d += levenshtein(decoding_string(b[i], string_index), decoding_string(gru_r[i], string_index))
        
        print('attention model levenshtein distance :', att_d/len(b))
        print('gru model levenshtein distance :', gru_d/len(b))

    elif sys.argv[1] == 'train_att':
        a, b = make_train_data()
        print(np.array(a).shape, np.array(b).shape)
        model = attention_model('train')
        
        for bb in b:
            if len(bb) != TIME_STEPS:
                print(len(bb), decoding_string(bb, string_index))

        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        ld = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(np.array(a)[::-1], np.array(b)[::-1], batch_size=128, epochs=300, verbose=1, validation_split=0.2, callbacks=[callback, ld], shuffle=True)
        model.save_weights('attention_test.h5')

        model.load_weights('attention_test.h5')
        # print(model.predict_on_batch([[a[-1]]]).shape)
        index = np.random.randint(len(a))
        print('input :', decoding_embeding(a[index], string_index))
        print('prdict :', decoding_string(model.predict_on_batch(np.array(a[index:index+1]))[0], string_index))
        print('answer :', decoding_string(b[index], string_index))
    
    elif sys.argv[1] == 'train_gru':
        model = attention_model('train', False)
        a, b = make_train_data()
        for bb in b:
            if len(bb) != TIME_STEPS:
                print(len(bb), decoding_string(bb, string_index))
        
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        ld = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(np.array(a)[::-1], np.array(b)[::-1], batch_size=128, epochs=300, verbose=1, validation_split=0.2, callbacks=[callback, ld], shuffle=True)
        model.save_weights('gru_test.h5')

        model.load_weights('gru_test.h5')
        # print(model.predict_on_batch([[a[-1]]]).shape)
        index = np.random.randint(len(a))
        print('input :', decoding_embeding(a[index], string_index))
        print('prdict :', decoding_string(model.predict_on_batch(np.array(a[index:index+1]))[0], string_index))
        print('answer :', decoding_string(b[index], string_index))