import tensorflow as tf

def create_CNN(widths, zero_pad=False):
    model = tf.keras.Sequential()
    # first convolutional layer
    model.add(tf.keras.layers.Conv2D(widths[0], kernel_size=4, activation='relu'))

    # next convolutional layers - inc stride size to 2
    for i in range(1, len(widths)):
        if zero_pad:
            model.add(tf.keras.layers.ZeroPadding2D())
        model.add(tf.keras.layers.Conv2D(widths[i], kernel_size=4, strides=2, activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model
