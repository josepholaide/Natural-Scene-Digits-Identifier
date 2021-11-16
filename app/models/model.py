from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout

def get_newcnn_model():
    model = Sequential([
        Conv2D(filters=128, input_shape=(32, 32, 3), kernel_size=(3, 3), padding='SAME',
               activation='relu', name='conv_1'),
        BatchNormalization(),  # <- Batch normalisation layer
        Dropout(0.25),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='SAME', name='conv_2'),
        MaxPooling2D(pool_size=(2, 2), name='pool_1'),
        BatchNormalization(),  # <- Batch normalisation layer
        Dropout(0.25),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='SAME', name='conv_3'),
        MaxPooling2D(pool_size=(2, 2), name='pool_2'),
        BatchNormalization(),  # <- Batch normalisation layer
        Dropout(0.25),
        Flatten(name='flatten'),
        Dense(units=256, activation='relu', name='dense_1'),
        Dropout(0.5),
        Dense(units=10, activation='softmax', name='dense_2')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_new_model():
    model = Sequential([
        Flatten(input_shape=(32, 32, 3), name='flatten'),
        Dense(units=1024, activation='relu', name='dense_1'),
        Dense(units=512, activation='relu', name='dense_2'),
        Dense(units=256, activation='relu', name='dense_3'),
        Dense(units=128, activation='relu', name='dense_4'),
        Dense(units=10, activation='softmax', name='dense_5')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model