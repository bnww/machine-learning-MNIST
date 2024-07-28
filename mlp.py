import tensorflow as tf

def create_model(nb_hidden_layers, layer_width, dropout_rate=0):
    model = tf.keras.Sequential()
    for i in range(nb_hidden_layers):
        model.add(tf.keras.layers.Dense(layer_width, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model
