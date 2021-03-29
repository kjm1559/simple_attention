from tensorflow.keras.layers import GRU, Input, Permute, Reshape, Dense, Multiply, dot, Lambda, TimeDistributed, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from source.utils import TIME_STEPS, string_index

def attention_mechanism(input):
    x = Dense(TIME_STEPS, activation='softmax')(input)
    return x

def attention_model(input_shape):
    input_ = Input(shape=(TIME_STEPS, input_shape,), name='input_data')
    gru_out = GRU(256, return_sequences=True, name='encode_gru')(input_)
    print('gru', gru_out.shape)
    attention_x = attention_mechanism(gru_out)
    gru_out = Permute((2, 1))(gru_out)
    attention_mul = K.batch_dot(gru_out, attention_x)
    attention_mul = Permute((2, 1))(attention_mul)
    output = GRU(input_shape, return_sequences=True, name='decode_gru')(K.reverse(attention_mul, axes=1))
    output = TimeDistributed(Dense(128, activation='relu'))(output)
    output = TimeDistributed(Dropout(0.2))(output)

    output = TimeDistributed(Dense(64, activation='relu'))(output)
    output = TimeDistributed(Dense(len(string_index), activation='softmax'))(output)
    # if sys.argv[1] == 'train':
    model = Model(inputs=input_, outputs=output)
    # elif sys.argv[1] == 'test':
    #     model = Model(inputs=input_, outputs=[output, attention_x])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model