# resnet model
import keras

#change
# from tensorflow.keras.mixed_precision import Policy
# from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split


import os
import tensorflow as tf
# import torch


import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration
# #change
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'





class Classifier_INCEPTION:

    #change batch
    def __init__(self, output_directory, input_shape, nb_classes, verbose=True, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            #change
            self.model.save_weights(os.path.join(self.output_directory, 'model_init.weights.h5'))
            # self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)


        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):


            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        print('compile model.....')

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
        print('end compile model....')

        #change put evreywhere val_loss instead of loss

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=17,
                                                      min_lr=0.0001)
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50)
        
        #change
        file_path = os.path.join(self.output_directory, 'best_model.hdf5')
        # file_path = self.output_directory + 'best_model.hdf5'
        

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        #change

        self.callbacks = [reduce_lr, early_stop, model_checkpoint]

        # self.callbacks = [reduce_lr, model_checkpoint]

        return model
    
    #change plot_test_acc to true
    #change x_val and y_val to x_test y_test
    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test, y_true, plot_test_acc=True):



        # #change
        # mixed_precision.set_global_policy('mixed_float16')

        
        #change
        print('GPU INFO')
        #change
        # print(torch.cuda.is_available())
        # print(torch.cuda.get_device_name())
        print(tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


        #change
        if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
            print('error no gpu')
            exit()

        # if len(keras.backend.tensorflow_backend._get_available_gpus()) == 0:
        #     print('error no gpu')
        #     exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        print('time staring....')

        start_time = time.time()

        if plot_test_acc:
            print('plot true')

            #change verbose to 1
            #change validation data to validation split

            # hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
            #                       verbose=1, validation_data=(x_val, y_val), callbacks=self.callbacks)

            #change :not used validation split to ensure balance

            # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
            #change
            with tf.device('CPU'):
                X_tensor_tr = tf.convert_to_tensor(x_train)
                Y_tensor_tr = tf.convert_to_tensor(y_train)
                X_tensor_v = tf.convert_to_tensor(x_val)
                Y_tensor_v = tf.convert_to_tensor(y_val)

            hist = self.model.fit(X_tensor_tr, Y_tensor_tr, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                verbose=1,  validation_data=(X_tensor_v, Y_tensor_v), callbacks=self.callbacks)
            
            # hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
            #                     verbose=1,  validation_data=(x_val, y_val), callbacks=self.callbacks)

            # hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
            #                      verbose=1, callbacks=self.callbacks)

            # hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
            #                   verbose=1, validation_split=0.2, callbacks=self.callbacks) 
        else:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=1, callbacks=self.callbacks)

        duration = time.time() - start_time

        print('fit time',duration)

        self.model.save(self.output_directory + 'last_model.hdf5')

    
        y_pred = self.predict(x_test, y_true, x_train, y_train, y_test,
                              return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration,
                               plot_test_acc=plot_test_acc)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred

