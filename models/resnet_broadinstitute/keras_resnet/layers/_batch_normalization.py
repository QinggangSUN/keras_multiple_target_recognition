import tensorflow as tf
if tf.__version__ >= '2.0':
    import tensorflow.keras.layers

    class BatchNormalization(tensorflow.keras.layers.BatchNormalization):
        """
        Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
        """

        def __init__(self, freeze, *args, **kwargs):
            self.freeze = freeze
            super(BatchNormalization, self).__init__(*args, **kwargs)

            # set to non-trainable if freeze is true
            self.trainable = not self.freeze

        def call(self, *args, **kwargs):
            # Force test mode if frozen, otherwise use default behaviour (i.e., training=None).
            if self.freeze:
                kwargs['training'] = False
            return super(BatchNormalization, self).call(*args, **kwargs)

            # dongwang218/keras-resnet
            # # return super.call, but set training
            # return super(BatchNormalization, self).call(training=(not self.freeze), *args, **kwargs)

        def get_config(self):
            config = super(BatchNormalization, self).get_config()
            config.update({'freeze': self.freeze})
            return config

else:
    import keras

    class BatchNormalization(keras.layers.BatchNormalization):
        """
        Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
        """

        def __init__(self, freeze, *args, **kwargs):
            self.freeze = freeze
            super(BatchNormalization, self).__init__(*args, **kwargs)

            # set to non-trainable if freeze is true
            self.trainable = not self.freeze

        def call(self, *args, **kwargs):
            # Force test mode if frozen, otherwise use default behaviour (i.e., training=None).
            if self.freeze:
                kwargs['training'] = False
            return super(BatchNormalization, self).call(*args, **kwargs)

            # dongwang218/keras-resnet
            # # return super.call, but set training
            # return super(BatchNormalization, self).call(training=(not self.freeze), *args, **kwargs)

        def get_config(self):
            config = super(BatchNormalization, self).get_config()
            config.update({'freeze': self.freeze})
            return config
