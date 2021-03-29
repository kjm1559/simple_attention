from source.utils import make_train_data, TIME_STEPS, decoding_string, encoding_string, string_index
from source.models import attention_model
import numpy as np

if __name__ == '__main__':
    X_train, y_train = make_train_data()
    for y in y_train:
        if len(y) != TIME_STEPS:
            print(len(y), decoding_string(y, string_index))

    model = attention_model(len(string_index))

    # epoch 50 
    for i in range(50):
        model.fit(np.array(X_train), np.array(y_train), batch_size=32, epochs=1, verbose=1, validation_split=0.2)
        model.save_weights('attention_test.h5')

        model.load_weights('attention_test.h5')
        index = np.random.randint(len(X_train))
        print('epoch :', i, 'input :', decoding_string(X_train[index], string_index))
        print('predict :', decoding_string(model.predict_on_batch(np.array([X_train[index]]))[0], string_index))
        print('answer  :', decoding_string(y_train[index], string_index))