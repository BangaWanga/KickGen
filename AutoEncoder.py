from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import logging


class AutoEncoder:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    @staticmethod
    def model0(input_shape: tuple):
        input_layer = Input(shape=input_shape)
        enc = Conv2D(8, (2, 2), activation='relu', name='E1')(input_layer)
        enc = MaxPooling2D((2, 2), padding='same', name='E2')(enc)
        enc = Conv2D(16, (2, 2), activation='relu', name='E3')(enc)
        enc = MaxPooling2D((2, 2), padding='same', name='E4')(enc)
        enc = Conv2D(8, (3, 3), activation='relu', name='E5')(enc)
        enc = MaxPooling2D((2, 2), padding='same', name='E6')(enc)
        encoder = Model(input_layer, enc)

        dec = Conv2D(8, (3, 3), activation='relu', padding='same', name='D1')(enc)
        dec = UpSampling2D((2, 2), name='D2')(dec)
        dec = Conv2D(8, (2, 2), activation='relu', padding='same', name='D3')(dec)
        dec = UpSampling2D((2, 2), name='D4')(dec)
        dec = Conv2D(64, (4, 4), activation='relu', padding='same', name='D5')(dec)
        dec = Conv2D(64, (3, 3), activation='relu', padding='same', name='D6')(dec)
        dec = Conv2D(32, (2, 2), activation='relu', padding='same', name='D7')(dec)
        dec = Conv2D(1, (2, 2), activation='sigmoid', padding='same', name='D9')(dec)
        dec = UpSampling2D((2, 2), name='D8')(dec)
        # decoder = Model(input_layer, dec)
        decoder = None
        autoencoder = Model(input_layer, dec)
        return encoder, decoder, autoencoder

    @staticmethod
    def model1(input_shape):
        input_img = Input(shape=input_shape)
        # Encoder
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(input_img)
        encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
        encoder_output = Conv2D(128, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(encoder_output)
        encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
        encoder_output = Conv2D(256, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(encoder_output)
        encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
        encoder_output = Conv2D(256, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(encoder_output)
        encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
        encoder_output = Conv2D(256, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder = Model(input_img, encoder_output)

        # Decoder
        decoder_output = Conv2D(8, (1, 1), activation='relu', padding='same')(encoder_output)
        decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(64, (4, 4), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(32, (2, 2), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(1, (3, 3), activation='tanh', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)

        decoder = None  # ToDo: Make decoder callable
        auto_encoder = Model(input_img, decoder_output)
        return encoder, decoder, auto_encoder

    def compile(self, input_shape: tuple, learn_rate: float = 1e-03, decay_rate: float = 0.0):
        e, d, a = self.model1(input_shape)
        self.encoder = e
        self.decoder = d
        self.autoencoder = a
        opt = Adam(lr=learn_rate, decay=decay_rate)
        self.autoencoder.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        self.autoencoder.summary()

    def fit_gen(self, gen_train, gen_val, checkpoint_dir, batch_size: int = 100, epochs: int = 100, save_best=True,
                reload=False):
        callbacks = []
        if reload:
            self.autoencoder.load_weights(checkpoint_dir)
        if save_best:
            callbacks.extend([
                ModelCheckpoint(checkpoint_dir, monitor='val_loss', save_best_only=True, verbose=1),
                # EarlyStopping(
                #     # Stop training when `val_loss` is no longer improving
                #     monitor="val_loss",
                #     # "no longer improving" being defined as "no better than 1e-2 less"
                #     min_delta=1e-2,
                #     # "no longer improving" being further defined as "for at least 2 epochs"
                #     patience=2,
                #     verbose=1,
                # )
            ])
        val_data = next(gen_val)
        for i in range(int(epochs/2)):
            train_data = next(gen_train)

            _history = self.autoencoder.fit(
                *train_data, validation_data=val_data,
                epochs=20, callbacks=callbacks,
                batch_size=batch_size
            )
        return

    def encode_decode_batch(self, sample: np.ndarray):
        return self.autoencoder.predict_on_batch(sample)

    def encode(self) -> np.ndarray:
        raise NotImplementedError

    def decode(self) -> np.ndarray:
        raise NotImplementedError

    def save(self, filepath):
        self.autoencoder.save(filepath)

    def load(self, filepath):
        self.autoencoder = load_model(filepath=filepath)


if __name__ == '__main__':
    auto = AutoEncoder()
    auto.compile(input_shape=(64, 128, 1))

