import tensorflow as tf

weight_path = "/local/datasets/idai720/checkpoint/vgg_face_weights.h5"

class VGG_Pre:
    def __init__(self, start_size = 64, input_shape = (224, 224, 3)):
        base_model = tf.keras.models.Sequential()
        base_model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                   input_shape=input_shape))
        base_model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                   input_shape=input_shape))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        base_model.add(
            tf.keras.layers.Conv2D(start_size*2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        base_model.add(
            tf.keras.layers.Conv2D(start_size*4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        base_model.add(
            tf.keras.layers.Conv2D(4096, kernel_size=(7, 7), strides=(1, 1), padding='valid',
                                   activation='relu'))
        base_model.add(tf.keras.layers.Dropout(0.5))
        base_model.add(
            tf.keras.layers.Conv2D(4096, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   activation='relu'))
        base_model.add(tf.keras.layers.Dropout(0.5))
        base_model.add(
            tf.keras.layers.Conv2D(2622, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   activation='relu'))

        base_model.add(tf.keras.layers.Flatten())
        base_model.add(tf.keras.layers.Activation('softmax'))
        base_model.load_weights(weight_path)


        base_model_output = tf.keras.layers.Flatten()(base_model.layers[-4].output)
        base_model_output = tf.keras.layers.Dense(256, activation='relu')(base_model_output)
        base_model_output = tf.keras.layers.Dense(1)(base_model_output)

        self.model = tf.keras.Model(inputs=base_model.input, outputs=base_model_output)
        self.model.compile(loss=tf.keras.losses.Huber(), metrics=['mae'], optimizer='SGD')

    def fit(self, X, y, X_val, y_val, sample_weight=None):
        # pre-trained weights of vgg-face model.
        # you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
        # related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, mode='auto',
                                                         min_lr=5e-5)

        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint/attractiveness.keras'
                                                          , monitor="val_loss", verbose=1
                                                          , save_best_only=True, mode='auto'
                                                          )
        self.model.fit(X, y, sample_weight=sample_weight, callbacks=[lr_reduce, checkpointer],
                                 validation_data=(X_val, y_val), batch_size=10, epochs=50)
        self.load_model('checkpoint/attractiveness.keras')


    def predict(self, X):
        return self.decision_function(X)

    def decision_function(self, X):
        pred = self.model.predict(X)
        return pred

    def load_model(self, checkpoint_filepath):
        self.model = tf.keras.models.load_model(checkpoint_filepath)


