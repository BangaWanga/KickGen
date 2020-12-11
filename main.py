import os

SR = 44100

if __name__ == '__main__':
    checkpoint_path = "models/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    sample_shape = (16, 18)     # imaginary shape for single spectrogram
    from AutoEncoder import AutoEncoder
    ae = AutoEncoder()
    ae.compile(input_shape=sample_shape)

    gen_train = []
    gen_val = []
    _history = ae.fit_gen(gen_train, gen_val, checkpoint_dir, epochs=100, batch_size=batch_size)



    # Instead of gen_train and gen_val (which are generators)
    # we could also use lists or numpy arrays.
    #
    # Example for python-generator:
    #
    # def gen():
    #     i=0
    #     while True:
    #         i += 1
    #         yield i