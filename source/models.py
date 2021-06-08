from tensorflow.keras.layers import Input, Permute, Dense, TimeDistributed, LSTM, Masking, Embedding
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from source.utils import TIME_STEPS, string_index

def attention_mechanism(inputs):
    x = Dense(TIME_STEPS, activation='softmax')(inputs)
    return x

def attention_model(type='train', flag=True):
    inputs = Input(shape=(TIME_STEPS,), name='input_data')
    # x = Masking()(inputs)
    x = Embedding(input_dim=41, output_dim=8)(inputs)
    gru_out = LSTM(128, return_sequences=True, name='encode_gru')(x)
    if flag:
        attention_x = attention_mechanism(gru_out)
        attention_mul = Permute((2, 1))(K.batch_dot(Permute((2, 1))(gru_out), attention_x))
    else:
        attention_mul = gru_out
    x = LSTM(128, return_sequences=True, name='decode_gru')(K.reverse(attention_mul, axes=1)) 
    output = TimeDistributed(Dense(len(string_index), activation='softmax'))(x)
    if type == 'test':
        model = Model(inputs=inputs, outputs=[output, attention_x])
    else:
        model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=masking_loss, metrics=[masking_acc])
    model.summary()
    return model

def masking_loss(y_true, y_pred):
    # return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.keras.losses.categorical_crossentropy(y_true[K.argmax(y_true) != 0], y_pred[K.argmax(y_true) != 0])


def masking_acc(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true[K.argmax(y_true) != 0], y_pred[K.argmax(y_true) != 0])