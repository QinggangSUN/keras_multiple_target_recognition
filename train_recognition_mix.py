# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:22:00 2018

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=too-many-locals, too-many-arguments, redefined-outer-name, unused-import

if __name__ == '__main__':
    import gc
    import json
    from keras import backend as K
    from keras import optimizers
    from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
    from keras.layers import Input
    from keras.models import load_model, Model
    from keras.utils import np_utils, plot_model, multi_gpu_model
    import keras_resnet
    import logging
    import numpy as np
    import os
    import tensorflow as tf

    from error import Error, ParameterError
    from file_operation import list_dirs, list_files_end_str, mkdir, walk_dirs_start_str
    from loss_acc import binary_acc, subset_acc_nhot, subset_acc_nhot_np
    from prepare_data_shipsear_recognition_mix_s0tos3 import PathSourceRoot, read_datas, save_datas
    from train_functions import output_history, save_model_struct, save_keras_model, load_keras_model

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)
    np.random.seed(1337)  # for reproducibility
#    # The below tf.set_random_seed() will make random number generation in the
#    # TensorFlow backend have a well-defined initial state. For further details,
#    # see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)
#    # Force TensorFlow to use single thread. Multiple threads are a potential
#    # source of non-reproducible results. For further details,
#    # see: https://stackoverflow.com/questions/42022950/
#    SESSION_CONF = tf.ConfigProto(
#        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    SESSION_CONF = tf.ConfigProto()
#    SESSION_CONF.intra_op_parallelism_threads = True
#    SESSION_CONF.inter_op_parallelism_threads = True
#
    # Limiting GPU memory growth (forbidden GPU OOM)
    SESSION_CONF.gpu_options.allow_growth = True

    SESS = tf.Session(graph=tf.get_default_graph(), config=SESSION_CONF)
    K.set_session(SESS)

    N_GPU = 2

    def create_model(x_list, y_list, num_model, n_gpu=1):
        """Create model for test model.
        Args:
            x_list (list[np.array]): input datasets.
            y_list (list[np.array]): output datasets.
            num_model (int): the number of model.
            n_gpu (int): number of gpu to train model.
        Returns:
            keras.Model: return model build.
        """
        d1 = x_list[1].shape[1]
        d2 = x_list[1].shape[2]
        od = y_list[1].shape[-1]

        if num_model == 90:
            model = build_model90(d1, d2, od)  # ResNet 1D 10
        elif num_model == 9:
            model = build_model9(d1, d2, od)  # ResNet 1D 18
        elif num_model == 6:
            model = build_model6(d1, d2, od)  # ResNet 1D 34
        elif num_model == 5:
            model = build_model5(d1, d2, od)  # ResNet 1D 50

        elif num_model == 120:
            model = build_model120(d1, d2, od)  # ResNet 2D 10
        elif num_model == 12:
            model = build_model12(d1, d2, od)  # ResNet 2D 18
        elif num_model == 13:
            model = build_model13(d1, d2, od)  # ResNet 2D 34
        elif num_model == 10:
            model = build_model10(d1, d2, od)  # ResNet 2D 50

        elif num_model == 150:
            model = build_model150(d1, d2, od)  # Complex ResNet 2D 10
        elif num_model == 15:
            model = build_model15(d1, d2, od)  # Complex ResNet 2D 18
        elif num_model == 16:
            model = build_model16(d1, d2, od)  # Complex ResNet 2D 34
        elif num_model == 17:
            model = build_model17(d1, d2, od)  # Complex ResNet 2D 50
        elif num_model == 18:
            model = build_model18(d1, d2, od)  # Complex DenseNet 2D 121
        elif num_model == 19:
            model = build_model19(d1, d2, od)  # Complex DenseNet 2D 169

        elif num_model == 200:
            model = build_model200(d1, od)  # Complex ResNet 1D 10
        elif num_model == 20:
            model = build_model20(d1, od)  # Complex ResNet 1D 18
        elif num_model == 21:
            model = build_model21(d1, od)  # Complex ResNet 1D 34
        elif num_model == 22:
            model = build_model22(d1, od)  # Complex ResNet 1D 50
        elif num_model == 23:
            model = build_model23(d1, od)  # Complex DenseNet 1D 121
        elif num_model == 24:
            model = build_model24(d1, od)  # Complex DenseNet 1D 169

        if num_model in list(range(15, 25))+[150, 200] and n_gpu > 1:
            inputs = Input(shape=(d1, d2, 2))
            target = model(inputs)
            model = Model(inputs, [target])

        return model

    def test_model(model, x_list, y_list, pbs, modelname, path_save, **kwargs):
        """Test predict and evaluate model.
        Args:
            model (keras.Model): keras model
            x_list (list[np.array]): input data sets
            y_list (list[np.array]): labels data sets
            pbs (int): predict bach size
            modelname (str): name of the model
            path_save (str): where to save predict results
        """

        bool_evaluate = True
        bool_subset_acc_nhot = True

        if 'evaluate' in kwargs.keys():
            bool_evaluate = kwargs['evaluate']
        if 'subset_acc_nhot' in kwargs.keys():
            bool_subset_acc_nhot = kwargs['subset_acc_nhot']

        subset_acc_name = kwargs['subset_acc_name'] if 'subset_acc_name' in kwargs.keys() else 'subset_acc_nhot'

        x_train, x_val, x_test = x_list
        y_train, y_val, y_test = y_list

        dict_r = dict()

        y_predict_train = np.array(model.predict(x_train, pbs), dtype=np.float32)
        y_predict_val = np.array(model.predict(x_val, pbs), dtype=np.float32)
        y_predict_test = np.array(model.predict(x_test, pbs), dtype=np.float32)

        dict_r.update({'p_train':y_predict_train.tolist()})
        dict_r.update({'l_train':y_train.tolist()})
        dict_r.update({'p_val':y_predict_val.tolist()})
        dict_r.update({'l_val':y_val.tolist()})
        dict_r.update({'p_test':y_predict_test.tolist()})
        dict_r.update({'l_test':y_test.tolist()})

        if bool_evaluate:
            score_keras_train = model.evaluate(x_train, y_train, verbose=0)  # batch_size=d0_train OOM
            score_keras_val = model.evaluate(x_val, y_val, verbose=0)
            score_keras_test = model.evaluate(x_test, y_test, verbose=0)

            dict_r.update({'loss_train':float(score_keras_train[0])})
            dict_r.update({'loss_val':float(score_keras_val[0])})
            dict_r.update({'loss_test':float(score_keras_test[0])})

            dict_r.update({'acc_train':float(score_keras_train[1])})
            dict_r.update({'acc_val':float(score_keras_val[1])})
            dict_r.update({'acc_test':float(score_keras_test[1])})

        if bool_subset_acc_nhot:
            d0_train = x_train.shape[0]  # d0_train = number of train samples
            d0_val = x_val.shape[0]  # d0_val = number of val samples
            d0_test = x_test.shape[0]  # d0_test = number of test samples

            subset_acc_nhot_train = subset_acc_nhot_np(y_train.reshape(d0_train, -1),
                                                       y_predict_train.reshape(d0_train, -1), 0.5)
            subset_acc_nhot_val = subset_acc_nhot_np(y_val.reshape(d0_val, -1),
                                                     y_predict_val.reshape(d0_val, -1), 0.5)
            subset_acc_nhot_test = subset_acc_nhot_np(y_test.reshape(d0_test, -1),
                                                      y_predict_test.reshape(d0_test, -1), 0.5)

            dict_r.update({f'{subset_acc_name}_train':float(subset_acc_nhot_train)})
            dict_r.update({f'{subset_acc_name}_val':float(subset_acc_nhot_val)})
            dict_r.update({f'{subset_acc_name}_test':float(subset_acc_nhot_test)})


        mkdir(os.path.join(path_save, 'loss'))
        with open(os.path.join(path_save, 'loss', f'test_{modelname}.json'), 'w', encoding='utf-8') as f_w:
            json.dump(dict_r, f_w)
        save_datas(dict_r, os.path.join(path_save, 'loss'), file_name=f'test_{modelname}',
            mode_batch='one_file_no_chunk')

    def test_check_models(path_save_model, x_list, y_list, num_model, pbs=256, kw_model='.hdf5',
        mode_load=0, dict_model_load=None, **kwargs):
        """Predict check models with models under path_save_model.
        Args:
            path_save_model (str): path where model saved.
            x_list (list[np.array]): input data sets.
            y_list (list[np.array]): labels data sets.
            num_model (int): number of the model.
            pbs (int, optional): predict bach size. Defaults to 256.
            kw_model (str, optional): file type of the saved models. Defaults to '.hdf5'.
            mode_load (int, optional): mode of save and load model. Defaults to 0.
            dict_model_load (dict, optional): custom objects of model. Defaults to None.
        Example:
            path_save_model = os.path.join(PATH_SAVE_ROOT, 'magspectrum_10547_10547_or_rand',
                                           'model_5_1_4', 'model', '1_n3_1')
            test_check_models(path_save_model, X_LIST, Y_LIST, 5, dict_model_load={'subset_acc_nhot':subset_acc_nhot})
        """

        check_filenames = list_files_end_str(path_save_model, kw_model, False)

        if mode_load == 1:  # load model from .h5 whole model
            # NOT work, if tensorflow verion < 2.1.0 and using subclass model from keras.Model
            model_file_name = list_files_end_str(path_save_model, '.h5', False)[0]
            check_model = load_model(os.path.join(path_save_model, model_file_name),
                                     **{'custom_objects':dict_model_load})

        if mode_load == 2:  # load model struct by model_from_json
            # this way will not work, if tensorflow verion < 2.1.0 and using subclass model from keras.Model
            check_model = load_keras_model(os.path.join(path_save_model, 'auto_model_struct.json'),
                                           mode=1, **{'custom_objects':dict_model_load})

        if mode_load == 3:  # build a new model
            n_gpu = 1 if 'n_gpu' not in kwargs.keys() else kwargs['n_gpu']
            check_model = create_model(x_list, y_list, num_model, n_gpu)

        if mode_load in {2, 3}:
            optimizer=optimizers.Adam(lr=1e-3) if 'optimizer' not in kwargs.keys() else kwargs['optimizer']
            loss='binary_crossentropy' if 'loss' not in kwargs.keys() else kwargs['loss']
            metrics=['accuracy', subset_acc_nhot] if 'metrics' not in kwargs.keys() else kwargs['metrics']
            check_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        for check_filename_i in check_filenames:
            logging.debug(f'model_name: {os.path.join(path_save_model, check_filename_i)}')

            if mode_load == 0:  # save and load whole model from .hdf5 check models
                # NOT work, if tensorflow verion < 2.1.0 and using subclass model from keras.Model
                check_model = load_model(os.path.join(path_save_model, check_filename_i),
                                         **{'custom_objects':dict_model_load})

            if mode_load in {1, 2, 3}:  # load model from .hdf5 check models
                check_model.load_weights(os.path.join(path_save_model, check_filename_i))

            test_model(check_model, x_list, y_list, pbs, check_filename_i, path_save_model,
                       **{'subset_acc_name':kwargs['subset_acc_name']})

    def test_all_check_models(path_save_root, x_list, y_list,
                              pbs=256, kw_model='.hdf5', num_models=None, model_load=0,
                              dict_model_load=None, **kwargs):
        """Predict all check models in dirs under path_save_root.
        Args:
            path_save_root (str): path root of the saved models.
            x_list (list[np.array]): input data sets.
            y_list (list[np.array]): labels data sets.
            pbs (int, optional): predict bach size. Defaults to 256.
            kw_model (str, optional): file type of the saved models. Defaults to '.hdf5'.
            num_models (list[int], optional): numbers of models. Defaults to None.
            mode_load (int, optional): mode of save and load model. Defaults to 0.
            dict_model_load (dict, optional): custom objects of model. Defaults to None.
        """
        dir_save_models = walk_dirs_start_str(path_save_root, 'model_', full=False)
        for path_dir_i in dir_save_models:
            num_model_i = int(path_dir_i[len('model_'):].split('_')[0])
            if num_models is None or (num_models and num_model_i in num_models):
                path_dir_model = os.path.join(path_save_root, path_dir_i, 'model')
                path_dir_models = list_dirs(path_dir_model)
                # logging.debug(f'path_dir_models {path_dir_models}')
                for path_dir_models_j in path_dir_models:
                    test_check_models(path_dir_models_j, x_list, y_list, num_model_i, pbs, kw_model,
                                      model_load, dict_model_load, **kwargs)

    def train_model(model, x_list, y_list, paras, path_save):
        """Train model.
        Args:
            model (keras.Model): model to train.
            x_list (list[np.array]): input data sets.
            y_list (list[np.array]): labels data sets.
            paras (dict): dictionary of the parameters.
            path_save (str): path to save modles.
        """

        i = paras['i']
        j = paras['j']
        epochs = paras['epochs']
        batch_size = paras['batch_size']
        optimizer_type = paras['optimizer'] if 'optimizer' in paras.keys() else 'adam'
        pbs = paras['pbs'] if 'pbs' in paras.keys() else 256
        subset_acc_name = paras['subset_acc_name'] if 'subset_acc_name' in paras.keys() else 'subset_acc_nhot'

        x_train, x_val, _ = x_list
        y_train, y_val, _ = y_list

        strj = 'n'+str(j)[1:] if j < 0 else str(j)
        modelname = str(i)+'_'+strj+'_'+str(epochs)

        logging.info('start train model')
        learn_rate = i*(10**j)
        if optimizer_type == 'adam':
            optimizer = optimizers.Adam(lr=learn_rate)
        elif optimizer_type == 'sgd':
            optimizer = optimizers.SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)

        n_gpu = paras['n_gpu'] if 'n_gpu' in paras.keys() else 1
        if n_gpu > 1:
            model = multi_gpu_model(model, gpus=n_gpu)

        if subset_acc_name == 'subset_acc_nhot':
            metrics = ['accuracy', subset_acc_nhot]
        elif subset_acc_name == 'binary_acc':
            metrics = ['accuracy', binary_acc]

        model.compile(
            optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

        if model.inputs is None:
            logging.warning('model.inputs is None, save model by tf.saved_model.save instead of model.save')

        path_check = os.path.join(path_save, 'model', modelname)
        mkdir(path_check)
        # check_filename = os.path.join(path_save, 'model', 'weights_'+modelname+'_{epoch:02d}_{val_acc:.2f}.hdf5')
        # checkpoint = ModelCheckpoint(filepath=check_filename, monitor='val_acc', mode='auto',
        #                              verbose=1, period=1, save_best_only=True)

        if subset_acc_name == 'subset_acc_nhot':
            check_filename = os.path.join(path_check, f'weights_{modelname}'+'_{epoch:02d}_{val_subset_acc_nhot:.2f}.hdf5')
            checkpoint = ModelCheckpoint(filepath=check_filename, monitor='val_subset_acc_nhot', mode='max',
                                        verbose=1, period=1, save_best_only=True)
        elif subset_acc_name == 'binary_acc':
            check_filename = os.path.join(path_check, f'weights_{modelname}'+'_{epoch:02d}_{val_binary_acc:.2f}.hdf5')
            checkpoint = ModelCheckpoint(filepath=check_filename, monitor='val_binary_acc', mode='max',
                                        verbose=1, period=1, save_best_only=True)

        path_board = os.path.join(path_save, 'tensorbord')
        mkdir(path_board)
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_val, y_val),
            callbacks=[TensorBoard(log_dir=path_board), checkpoint])

        # list all data in history
        # logging.debug(history.history.keys())
        # summarize history for loss
        path_history = os.path.join(path_save, 'loss')
        mkdir(path_history)

        output_history(
            ['r', 'g'], [history.history['loss'], history.history['val_loss']],
            os.path.join(path_history, f'loss_{modelname}.svg'), show=False,
            title='loss val_loss', label_x='epoch', label_y='loss', loc='upper left')

        if subset_acc_name == 'subset_acc_nhot':
            output_history(
                ['b', 'y'], [history.history['subset_acc_nhot'], history.history[f'val_subset_acc_nhot']],
                os.path.join(path_history, f'subset_accuracy_{modelname}.svg'),
                show=False, title='subset_accuracy val_subset_accuracy',
                label_x='epoch', label_y='accuracy', loc='upper left')
        elif subset_acc_name == 'binary_acc':
            output_history(
                ['b', 'y'], [history.history['binary_acc'], history.history[f'val_binary_acc']],
                os.path.join(path_history, f'binary_accuracy_{modelname}.svg'),
                show=False, title='binary_accuracy val_binary_accuracy',
                label_x='epoch', label_y='accuracy', loc='upper left')

        output_history(
            ['b', 'y'], [history.history['acc'], history.history['val_acc']],
            os.path.join(path_history, f'accuracy_{modelname}.svg'),
            show=False, title='accuracy val_accuracy',
            label_x='epoch', label_y='accuracy', loc='upper left')

        path_save_model = os.path.join(path_save, 'model', modelname)
        mkdir(path_save_model)

        save_model_struct(model, path_save_model, 'auto_model_struct')
        model.save(os.path.join(path_save_model, modelname+'.h5'))

        test_model(model, x_list, y_list, pbs, modelname, path_save)

        if 'test_check_models' in paras.keys() and paras['test_check_models']:
            test_check_models(path_save_model, x_list, y_list,
                              paras['num_model'], pbs, paras['kw_model'],
                              paras['mode_load'], paras['dict_model_load'])

    def load_data(path_data=None, path_root=None, form_src='None', scaler_data='or', sub_set_way='rand', **kwargs):
        """Load data from files.
        Args:
            path_data (str): path of data, if this is not None, use this path
            path_root (str): if path_data is None, compute path_data through class PathSourceRoot
            form_src (str): type of feature
            scaler_data (str) ['or', 'mm']: way of scaler original data
            sub_set_way (str) ['rand', 'order']: way of split data sets
        Return:
            x_list (list[np.array]): list of feature data sets.
            y_list (list[np.array]): list of label data sets.
        """

        if path_data is None:
            if form_src == 'wav':
                path_class = PathSourceRoot(
                    path_root, form_src=form_src, scaler_data=scaler_data, sub_set_way=sub_set_way)
            elif form_src in {'magspectrum', 'angspectrum', 'realspectrum', 'imgspectrum'}:
                if 'win_length' in kwargs.keys():
                    win_length = kwargs['win_length']
                else:
                    raise ParameterError('Need keyword para "win_length"')

                if 'hop_length' in kwargs.keys():
                    hop_length = kwargs['hop_length']
                else:
                    raise ParameterError('Need keyword para "hop_length"')

                path_class = PathSourceRoot(
                    path_root, form_src=form_src, win_length=win_length, hop_length=hop_length,
                    scaler_data=scaler_data, sub_set_way=sub_set_way)
            elif form_src == 'logmelspectrum':
                if 'win_length' in kwargs.keys():
                    win_length = kwargs['win_length']
                else:
                    raise ParameterError('Need keyword para "win_length"')

                if 'hop_length' in kwargs.keys():
                    hop_length = kwargs['hop_length']
                else:
                    raise ParameterError('Need keyword para "hop_length"')

                if 'n_mels' in kwargs.keys():
                    n_mels = kwargs['n_mels']
                else:
                    raise ParameterError('Need keyword para "n_mels"')
                path_class = PathSourceRoot(
                    path_root, form_src=form_src,
                    win_length=win_length, hop_length=hop_length, n_mels=n_mels,
                    scaler_data=scaler_data, sub_set_way=sub_set_way)
            elif form_src == 'mfcc':
                if 'win_length' in kwargs.keys():
                    win_length = kwargs['win_length']
                else:
                    raise ParameterError('Need keyword para "win_length"')

                if 'hop_length' in kwargs.keys():
                    hop_length = kwargs['hop_length']
                else:
                    raise ParameterError('Need keyword para "hop_length"')

                if 'n_mels' in kwargs.keys():
                    n_mels = kwargs['n_mels']
                else:
                    raise ParameterError('Need keyword para "n_mels"')

                if 'n_mfcc' in kwargs.keys():
                    n_mfcc = kwargs['n_mfcc']
                else:
                    raise ParameterError('Need keyword para "n_mfcc"')

                path_class = PathSourceRoot(
                    path_root, form_src=form_src,
                    win_length=win_length, hop_length=hop_length,
                    n_mels=n_mels, n_mfcc=n_mfcc,
                    scaler_data=scaler_data, sub_set_way=sub_set_way)
            elif form_src == 'demon':
                if 'high' in kwargs.keys():
                    high = kwargs['high']
                else:
                    raise ParameterError('Need keyword para "high"')

                if 'low' in kwargs.keys():
                    low = kwargs['low']
                else:
                    raise ParameterError('Need keyword para "low"')

                if 'cutoff' in kwargs.keys():
                    cutoff = kwargs['cutoff']
                else:
                    raise ParameterError('Need keyword para "cutoff"')
                path_class = PathSourceRoot(
                    path_root, form_src=form_src,
                    high=high, low=low, cutoff=cutoff,
                    scaler_data=scaler_data, sub_set_way=sub_set_way)
            else:
                raise ParameterError('Invalid form_src')

            path_data = path_class.path_source

        x_list = read_datas(path_data, ['X_train', 'X_val', 'X_test'])
        y_list = read_datas(path_data, ['Y_train', 'Y_val', 'Y_test'])

        logging.info('data load finished')
        logging.debug('X shape')
        for x_i in x_list:
            logging.debug(x_i.shape)
        logging.debug('Y shape')
        for y_i in y_list:
            logging.debug(y_i.shape)

        return x_list, y_list

    def standar_data(x_list, y_list, dim_input, dim_output, min_input=32, test_few=False,
        one_out=False, n_one_out=0):
        """Standardize data shape for network input and output.
        Args:
            x_list (list[np.array]): input datasets.
            y_list (list[np.array]): output datasets.
            dim_input (int): dimension of the input data.
            dim_output (int): dimension of the output data.
            min_input (int, optional): minimum dimension of the input. Defaults to 32.
            test_few (bool, optional): return few data for test. Defaults to False.
            one_out (bool, optional): for one output s1~3. Defaults to False.
            n_one_out (int, optional): number of the output s1~3. Defaults to 0.
        Returns:
            x_list (list[np.array]): standardized input datasets.
            y_list (list[np.array]): standardized output datasets.
        """

        if test_few:  # only for test few samples
            x_list[0], y_list[0] = x_list[0][:6, :, :], y_list[0][:6, :, :]
            x_list[1], y_list[1] = x_list[1][:2, :, :], y_list[1][:2, :, :]
            x_list[2], y_list[2] = x_list[2][:2, :, :], y_list[2][:2, :, :]
        else:  # only for full data
            y_list = [np.asarray(y_i) for y_i in y_list]  # (n_samples, 1, od)

        if dim_input == 1:  # only for 1D network input
            x_list = [np.asarray(x_i).transpose(0, 2, 1) for x_i in x_list]  # (n_samples, fl, 1)
        elif dim_input == 2:  # only for 2D network input
            x_list = [np.expand_dims(x_i, -1) for x_i in x_list]  # (n_samples, t, fl, 1)

        if dim_output == 1:  # only for 1D network output
            y_list = [np.squeeze(y_i) for y_i in y_list]  # (n_samples, od)

        if one_out:
            # only for one output s1~3
            for i, y_i in enumerate(y_list):
                if np.rank(y_i) == 3:
                    y_list[i] = y_i[:, :, n_one_out]
                elif np.rank(y_i) == 2:
                    y_list[i] = y_i[:, n_one_out]

        if min_input > 0:
            if dim_input == 2:  # only for 2D input padding, input size must >= (32, 32, 1)
                d1 = x_list[1].shape[1]  # (n_samples, t, fl, 1)
                d2 = x_list[1].shape[2]
                if d1 < min_input:
                    x_list = [np.pad(x_i,
                                    ((0,0), (0,min_input-d1), (0,0), (0,0)),
                                    'constant', constant_values=(0,0)) for x_i in x_list]
                if d2 < min_input:
                    x_list = [np.pad(x_i,
                                    ((0,0), (0,0), (0,min_input-d2), (0,0)),
                                    'constant', constant_values=(0,0)) for x_i in x_list]

        logging.debug('X shape')
        for x_i in x_list:
            logging.debug(x_i.shape)
        logging.debug('Y shape')
        for y_i in y_list:
            logging.debug(y_i.shape)

        return x_list, y_list

    from models.models_recognition import build_model, build_model2, build_model3, build_model4
    from models.models_recognition import build_model90, build_model9, build_model6, build_model5
    from models.models_recognition import build_model7, build_model8
    from models.models_recognition import build_model120, build_model12, build_model13, build_model10
    from models.models_recognition import build_model11, build_model14
    from models.models_recognition import build_model150, build_model15, build_model16, build_model17
    from models.models_recognition import build_model18, build_model19
    from models.models_recognition import build_model200, build_model20, build_model21, build_model22
    from models.models_recognition import build_model23, build_model24

    def search_models(x_list, y_list, model_list, path_save, **kwargs):
        """train the models.
        Args:
            x_list (list[np.array]): input datasets.
            y_list (list[np.array]): output datasets.
            model_list (list[int]): numbers of the models to train.
            path_save (str): path to save models.
        """
        x_train, x_val, x_test = x_list  # (n_samples, fl, 1)
        y_train, y_val, y_test = y_list  # (n_samples, 1, od)

        d0_train = x_train.shape[0]  # d0_train = number of train samples
        d0_val = x_val.shape[0]  # d0_val = number of val samples
        d0_test = x_test.shape[0]  # d0_test = number of test samples
        d1 = x_val.shape[1]
        d2 = x_val.shape[2]
        od = y_test.shape[-1]

        i, j = kwargs['i'], kwargs['j']
        subset_acc_name = kwargs['subset_acc_name'] if 'subset_acc_name' in kwargs.keys() else 'subset_acc_nhot'

        if 90 in model_list:
            model90 = build_model90(d1, d2, od)  # ResNet 1D 10
            path_result = os.path.join(path_save, 'model_90_1_4')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':64, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model90, paras=paras, x_list=x_list, y_list=y_list, path_save=path_result)
            del model90
            gc.collect()

        if 9 in model_list:
            model9 = build_model9(d1, d2, od)  # ResNet 1D 18
            path_result = os.path.join(path_save, 'model_9_1_4')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':64, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model9, paras=paras, x_list=x_list, y_list=y_list, path_save=path_result)
            del model9
            gc.collect()

        if 6 in model_list:
            model6 = build_model6(d1, d2, od)  # ResNet 1D 34
            path_result = os.path.join(path_save, 'model_6_1_4')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':64, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model6, paras=paras, x_list=x_list, y_list=y_list, path_save=path_result)
            del model6
            gc.collect()

        if 5 in model_list:
            model5 = build_model5(d1, d2, od)  # ResNet 1D 50
            path_result = os.path.join(path_save, 'model_5_1_4')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':64, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model5, paras=paras, x_list=x_list, y_list=y_list, path_save=path_result)
            del model5
            gc.collect()

        if 7 in model_list:
            model7 = build_model7(d1, d2, od)  # DenseNet 1D 121
            path_result = os.path.join(path_save, 'model_7_1_3')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':64, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model7, paras=paras, x_list=x_list, y_list=y_list, path_save=path_result)
            del model7
            gc.collect()

        if 8 in model_list:
            model8 = build_model8(d1, d2, od)  # DenseNet 1D 169
            path_result = os.path.join(path_save, 'model_8_1_3')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':64, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model8, paras=paras, x_list=x_list, y_list=y_list, path_save=path_result)
            del model8
            gc.collect()

        if 120 in model_list:
            model120 = build_model120(d1, d2, od)  # ResNet 2D 10
            path_result = os.path.join(path_save, 'model_120_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':64, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model120, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model120
            gc.collect()

        if 12 in model_list:
            model12 = build_model12(d1, d2, od)  # ResNet 2D 18
            path_result = os.path.join(path_save, 'model_12_1_3')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'test_check_models':False, 'subset_acc_name':subset_acc_name}
            train_model(model=model12, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model12
            gc.collect()

        if 13 in model_list:
            model13 = build_model13(d1, d2, od)  # ResNet 2D 34
            path_result = os.path.join(path_save, 'model_13_1_3')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'test_check_models':False, 'subset_acc_name':subset_acc_name}
            train_model(model=model13, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model13
            gc.collect()

        if 10 in model_list:
            model10 = build_model10(d1, d2, od)  # ResNet 2D 50
            path_result = os.path.join(path_save, 'model_10_1_3')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model10, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model10
            gc.collect()

        if 11 in model_list:
            model11 = build_model11(d1, d2, od)  # DenseNet 2D 121
            path_result = os.path.join(path_save, 'model_11_1_3')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model11, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model11
            gc.collect()

        if 14 in model_list:
            model14 = build_model14(d1, d2, od)  # DenseNet 2D 169
            path_result = os.path.join(path_save, 'model_14_1_3')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model14, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model14
            gc.collect()

        if 150 in model_list:
            model150 = build_model150(d1, d2, od)  # Complex ResNet 2D 10
            path_result = os.path.join(path_save, 'model_150_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model150, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model150
            gc.collect()

        if 15 in model_list:
            model15 = build_model15(d1, d2, od)  # Complex ResNet 2D 18
            path_result = os.path.join(path_save, 'model_15_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model15, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model15
            gc.collect()

        if 16 in model_list:
            model16 = build_model16(d1, d2, od)  # Complex ResNet 2D 34
            path_result = os.path.join(path_save, 'model_16_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model16, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model16
            gc.collect()

        if 17 in model_list:
            model17 = build_model17(d1, d2, od)  # Complex ResNet 2D 50
            path_result = os.path.join(path_save, 'model_17_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model17, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model17
            gc.collect()

        if 18 in model_list:
            model18 = build_model18(d1, d2, od)  # Complex DenseNet 2D 121
            path_result = os.path.join(path_save, 'model_18_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model18, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model18
            gc.collect()

        if 19 in model_list:
            model19 = build_model19(d1, d2, od)  # Complex DenseNet 2D 169
            path_result = os.path.join(path_save, 'model_19_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model19, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model19
            gc.collect()

        if 200 in model_list:
            model200 = build_model200(d1, od)  # Complex ResNet 1D 10
            path_result = os.path.join(path_save, 'model_200_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model200, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model200
            gc.collect()

        if 20 in model_list:
            model20 = build_model20(d1, od)  # Complex ResNet 1D 18
            path_result = os.path.join(path_save, 'model_20_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model20, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model20
            gc.collect()

        if 21 in model_list:
            model21 = build_model21(d1, od)  # Complex ResNet 1D 34
            path_result = os.path.join(path_save, 'model_21_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model21, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model21
            gc.collect()

        if 22 in model_list:
            model22 = build_model22(d1, od)  # Complex ResNet 1D 50
            path_result = os.path.join(path_save, 'model_22_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model22, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model22
            gc.collect()

        if 23 in model_list:
            model23 = build_model23(d1, od)  # Complex DenseNet 1D 121
            path_result = os.path.join(path_save, 'model_23_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model23, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model23
            gc.collect()

        if 24 in model_list:
            model24 = build_model24(d1, od)  # Complex DenseNet 1D 169
            path_result = os.path.join(path_save, 'model_24_1_1')
            paras = {'i':i, 'j':j, 'epochs':100, 'batch_size':10, 'n_gpu':N_GPU, 'subset_acc_name':subset_acc_name}
            train_model(model=model24, x_list=x_list, y_list=y_list, paras=paras, path_save=path_result)
            del model24
            gc.collect()
    # ==========================================================================
    PATH_ROOT = '/home/sqg/data/shipsEar/mix_recognition'
    PATH_SAVE_ROOT = '../result_recognition'
    mkdir(PATH_SAVE_ROOT)
    # --------------------------------------------------------------------------
    from models.complex_networks_keras_tf1.models.resnet_models_2d import ResNet2D18 as ComplexResNet2D18
    from models.complex_networks_keras_tf1.models.resnet_models_2d import ResNet2D34 as ComplexResNet2D34
    from models.complex_networks_keras_tf1.models.resnet_models_2d import ResNet2D50 as ComplexResNet2D50
    from models.complex_networks_keras_tf1.models.densenet_models_2d import DenseNet2D121 as ComplexDenseNet2D121
    from models.complex_networks_keras_tf1.models.densenet_models_2d import DenseNet2D169 as ComplexDenseNet2D169
    from models.complex_networks_keras_tf1.models.resnet_models_1d import ResNet1D18 as ComplexResNet1D18
    from models.complex_networks_keras_tf1.models.resnet_models_1d import ResNet1D34 as ComplexResNet1D34
    from models.complex_networks_keras_tf1.models.resnet_models_1d import ResNet1D50 as ComplexResNet1D50
    from models.complex_networks_keras_tf1.models.densenet_models_1d import DenseNet1D121 as ComplexDenseNet1D121
    from models.complex_networks_keras_tf1.models.densenet_models_1d import DenseNet1D169 as ComplexDenseNet1D169
    DICT_MODEL_STRUCT = {'ResNet2D18':keras_resnet.models.ResNet2D18,
                         'ResNet2D34':keras_resnet.models.ResNet2D34,
                         'ResNet2D50':keras_resnet.models.ResNet2D50,
                         'ComplexResNet2D18':ComplexResNet2D18,
                         'ComplexResNet2D34':ComplexResNet2D34,
                         'ComplexResNet2D50':ComplexResNet2D50,
                         'ComplexDenseNet2D121':ComplexDenseNet2D121,
                         'ComplexDenseNet2D169':ComplexDenseNet2D169,
                         'ComplexResNet1D18':ComplexResNet1D18,
                         'ComplexResNet1D34':ComplexResNet1D34,
                         'ComplexResNet1D50':ComplexResNet1D50,
                         'ComplexDenseNet1D121':ComplexDenseNet1D121,
                         'ComplexDenseNet1D169':ComplexDenseNet1D169,
                         'BatchNormalization':keras_resnet.layers.BatchNormalization}
    # --------------------------------------------------------------------------
    subset_acc_name = 'binary_acc'
    if subset_acc_name == 'subset_acc_nhot':
        DICT_MODEL_CONFIG = {'subset_acc_nhot':subset_acc_nhot}
        DICT_MODEL_COMPILE = {'optimizer':optimizers.Adam(lr=1e-3),
                              'loss':'binary_crossentropy',
                              'metrics':['accuracy', subset_acc_nhot],
                              'subset_acc_name':subset_acc_name,
                              'n_gpu':N_GPU}
    elif subset_acc_name == 'binary_acc':
        # only for being compatible with old version, which named 'binary_acc' instead of 'subset_acc_nhot'
        DICT_MODEL_CONFIG = {'binary_acc':binary_acc}
        DICT_MODEL_COMPILE = {'optimizer':optimizers.Adam(lr=1e-3),
                              'loss':'binary_crossentropy',
                              'metrics':['accuracy', binary_acc],
                              'subset_acc_name':subset_acc_name,
                              'n_gpu':N_GPU}
    # --------------------------------------------------------------------------
    # for wavmat
    path_save = os.path.join(PATH_SAVE_ROOT, 'wavmat_or_rand')  # _mm_order
    mkdir(path_save)
    x_list, y_list = load_data(path_root=PATH_ROOT, form_src='wav', scaler_data='or', sub_set_way='rand')
    x_list, y_list = standar_data(x_list, y_list, 1, 2, test_few=False)

    for i in range(1, 2):
        for j in range(-3, -4, -1):
            search_models(x_list, y_list, [7, 8], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

    test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[7, 8], model_load=0,
                            dict_model_load=DICT_MODEL_CONFIG, **{'subset_acc_name':subset_acc_name})
#    test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[7, 8], model_load=3,
#                          dict_model_load={**DICT_MODEL_CONFIG, **DICT_MODEL_STRUCT},
#                          kw_model='.hdf5',
#                          **DICT_MODEL_COMPILE)

    y_list = [np.squeeze(y_i) for y_i in y_list]  # (n_samples, od)

    for i in range(1, 2):
        for j in range(-3, -4, -1):
            search_models(x_list, y_list, [90, 9, 6, 5], path_save, **{'i':i, 'j':j})

    test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[90, 9, 6, 5], model_load=3,
                          dict_model_load={**DICT_MODEL_CONFIG, **DICT_MODEL_STRUCT},
                          kw_model='.hdf5',
                          **DICT_MODEL_COMPILE)
    # --------------------------------------------------------------------------
    # for 1D magspectrum
    WIN_LENGTH = 10547
    HOP_LENGTH = 10547
    path_save = os.path.join(PATH_SAVE_ROOT, f'magspectrum_{WIN_LENGTH}_{HOP_LENGTH}_or_rand')
    mkdir(path_save)
    x_list, y_list = load_data(path_root=PATH_ROOT, form_src='magspectrum', scaler_data='or', sub_set_way='rand',
                               **{'win_length':WIN_LENGTH, 'hop_length':HOP_LENGTH})
    x_list, y_list = standar_data(x_list, y_list, 1, 2, test_few=False)

    for i in range(1, 2):
        for j in range(-3, -4, -1):
            search_models(x_list, y_list, [7, 8], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

    test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[7, 8], model_load=0,
                            dict_model_load=DICT_MODEL_CONFIG, **{'subset_acc_name':subset_acc_name})


    y_list = [np.squeeze(y_i) for y_i in y_list]  # (n_samples, od)
    for i in range(1, 2):
        for j in range(-3, -4, -1):
            search_models(x_list, y_list, [9, 6, 5], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

    test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[9, 6, 5], model_load=3,
                          dict_model_load={**DICT_MODEL_CONFIG, **DICT_MODEL_STRUCT},
                          kw_model='.hdf5',
                          **DICT_MODEL_COMPILE)
    # --------------------------------------------------------------------------
    # for 2D magspectrum
    WIN_LIST = [264, 528, 1056, 1582, 2110, 2638, 3164]
    HOP_LIST = [ 66, 132,  264,  396,  527,  659,  791]

    for win_i, hop_i in zip(WIN_LIST, HOP_LIST):
        path_save = os.path.join(PATH_SAVE_ROOT, f'magspectrum_{win_i}_{hop_i}_or_rand')
        mkdir(path_save)

        x_list, y_list = load_data(path_root=PATH_ROOT, form_src='magspectrum', scaler_data='or', sub_set_way='rand',
                                   **{'win_length':win_i, 'hop_length':hop_i})
        x_list, y_list = standar_data(x_list, y_list, 2, 2, test_few=False)

        for i in range(1, 2):
            for j in range(-3, -4, -1):
                search_models(x_list, y_list, [11, 14], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

        test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[11, 14], model_load=0,
                              dict_model_load=DICT_MODEL_CONFIG, **{'subset_acc_name':subset_acc_name})

        y_list = [np.squeeze(y_i) for y_i in y_list]  # (n_samples, od)

        for i in range(1, 2):
            for j in range(-3, -4, -1):
                search_models(x_list, y_list, [120, 12, 13, 10], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

        test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[120, 12, 13, 10], model_load=3,
                              dict_model_load={**DICT_MODEL_CONFIG, **DICT_MODEL_STRUCT},
                              kw_model='.hdf5',
                              **DICT_MODEL_COMPILE)
    # --------------------------------------------------------------------------
    # for log-mel spectrum
    WIN_LIST = [264, 528, 1056, 1582, 2110, 2638, 3164]
    HOP_LIST = [ 66, 132,  264,  396,  527,  659,  791]

    N_MELS = [512, 256, 128]

    for win_i, hop_i in zip(WIN_LIST, HOP_LIST):
        for n_mels_i in N_MELS:
            path_save = os.path.join(PATH_SAVE_ROOT, f'logmelspectrum_{win_i}_{hop_i}_{n_mels_i}_or_rand')
            mkdir(path_save)
            x_list, y_list = load_data(path_root=PATH_ROOT, form_src='logmelspectrum', scaler_data='or', sub_set_way='rand',
                                **{'win_length':win_i, 'hop_length':hop_i, 'n_mels':n_mels_i})
            x_list, y_list = standar_data(x_list, y_list, 2, 2)

            for i in range(1, 2):
                for j in range(-3, -4, -1):
                    search_models(x_list, y_list, [11, 14], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

            test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[11, 14], model_load=0,
                                dict_model_load=DICT_MODEL_CONFIG, **{'subset_acc_name':subset_acc_name})

            y_list = [np.squeeze(y_i) for y_i in y_list]  # (n_samples, od)
            for i in range(1, 2):
                for j in range(-3, -4, -1):
                    search_models(x_list, y_list, [120, 12, 13, 10], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

            test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[120, 12, 13, 10], model_load=3,
                                dict_model_load={**DICT_MODEL_CONFIG, **DICT_MODEL_STRUCT},
                                kw_model='.hdf5',
                                **DICT_MODEL_COMPILE)
    # --------------------------------------------------------------------------
    #  for mfcc
    WIN_LIST = [264, 528, 1056, 1582, 2110, 2638, 3164]
    HOP_LIST = [ 66, 132,  264,  396,  527,  659,  791]

    N_MELS = [512, 256, 128]

    N_MFCCS = [80, 40, 20]

    for win_i, hop_i in zip(WIN_LIST, HOP_LIST):
        for n_mels_i in N_MELS:
            for n_mfcc_i in N_MFCCS:
                path_save = os.path.join(PATH_SAVE_ROOT, f'mfcc_{win_i}_{hop_i}_{n_mels_i}_{n_mfcc_i}_or_rand')
                mkdir(path_save)
                x_list, y_list = load_data(path_root=PATH_ROOT, form_src='mfcc', scaler_data='or', sub_set_way='rand',
                                            **{'win_length':win_i, 'hop_length':hop_i, 'n_mels':n_mels_i, 'n_mfcc':n_mfcc_i})
                x_list, y_list = standar_data(x_list, y_list, 2, 2, test_few=False)

                for i in range(1, 2):
                    for j in range(-3, -4, -1):
                        search_models(x_list, y_list, [11, 14], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

                test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[11, 14], model_load=0,
                                    dict_model_load=DICT_MODEL_CONFIG, **{'subset_acc_name':subset_acc_name})

                y_list = [np.squeeze(y_i) for y_i in y_list]  # (n_samples, od)
                for i in range(1, 2):
                    for j in range(-3, -4, -1):
                        search_models(x_list, y_list, [120, 12, 13, 10], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

                test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[120, 12, 13, 10], model_load=3,
                                      dict_model_load={**DICT_MODEL_CONFIG, **DICT_MODEL_STRUCT},
                                      kw_model='.hdf5',
                                      **DICT_MODEL_COMPILE)
    # --------------------------------------------------------------------------
    # for demon feature
    HIGH_LIST = [7910.1]
    LOW_LIST = [5273.4]
    CUTOFF_LIST = [1000]

    for high_i, low_i in zip(HIGH_LIST, LOW_LIST):
        for cutoff_i in CUTOFF_LIST:
            path_save = os.path.join(PATH_SAVE_ROOT, f'demon_{high_i}_{low_i}_{cutoff_i}_or_rand')
            mkdir(path_save)
            x_list, y_list = load_data(path_root=PATH_ROOT, form_src='demon', scaler_data='or', sub_set_way='rand',
                                        **{'high':high_i, 'low':low_i, 'cutoff':cutoff_i})
            x_list, y_list = standar_data(x_list, y_list, 1, 2, test_few=False)

            for i in range(1, 2):
                for j in range(-3, -4, -1):
                    search_models(x_list, y_list, [7, 8], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

            test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[7, 8], model_load=0,
                                    dict_model_load=DICT_MODEL_CONFIG, **{'subset_acc_name':subset_acc_name})

            y_list = [np.squeeze(y_i) for y_i in y_list]  # (n_samples, od)
            for i in range(1, 2):
                for j in range(-3, -4, -1):
                    search_models(x_list, y_list, [90, 9, 6, 5], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

            test_all_check_models(path_save, x_list=x_list, y_list=y_list, num_models=[90, 9, 6, 5], model_load=3,
                                  dict_model_load={**DICT_MODEL_CONFIG, **DICT_MODEL_STRUCT},
                                  kw_model='.hdf5',
                                  **DICT_MODEL_COMPILE)
    # --------------------------------------------------------------------------
    # for 2D realspectrum and imgspectrum
    WIN_LIST = [264, 528, 1056, 1582, 2110, 2638, 3164]
    HOP_LIST = [ 66, 132,  264,  396,  527,  659,  791]

    for win_i, hop_i in zip(WIN_LIST, HOP_LIST):
        path_save = os.path.join(PATH_SAVE_ROOT, f'real_img_spectrum_{win_i}_{hop_i}_or_rand')
        mkdir(path_save)
        real_x_list, y_list = load_data(path_root=PATH_ROOT, form_src='realspectrum', scaler_data='or', sub_set_way='rand',
                                        **{'win_length':win_i, 'hop_length':hop_i})
        real_x_list, y_list = standar_data(real_x_list, y_list, 2, 1, test_few=False)

        img_x_list, y_list = load_data(path_root=PATH_ROOT, form_src='imgspectrum', scaler_data='or', sub_set_way='rand',
                                        **{'win_length':win_i, 'hop_length':hop_i})
        img_x_list, y_list = standar_data(img_x_list, y_list, 2, 1, test_few=False)

        x_list = [np.concatenate([real_i, img_i], axis=-1) for real_i, img_i in zip(real_x_list, img_x_list)]

        for i in range(1, 2):
            for j in range(-3, -4, -1):
                search_models(x_list, y_list, [150, 15, 16, 17, 18, 19], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

        test_all_check_models(path_save, x_list=x_list, y_list=y_list,
                                num_models=[150, 15, 16, 17, 18, 19], model_load=3,
                                dict_model_load={**DICT_MODEL_CONFIG, **DICT_MODEL_STRUCT},
                                kw_model='.hdf5',
                                **DICT_MODEL_COMPILE)
    # --------------------------------------------------------------------------
    # for 1D realspectrum and imgspectrum
    WIN_LIST = [10547]
    HOP_LIST = [10547]

    for win_i, hop_i in zip(WIN_LIST, HOP_LIST):
        path_save = os.path.join(PATH_SAVE_ROOT, f'real_img_spectrum_{win_i}_{hop_i}_or_rand')
        mkdir(path_save)
        real_x_list, y_list = load_data(path_root=PATH_ROOT, form_src='realspectrum', scaler_data='or', sub_set_way='rand',
                                        **{'win_length':win_i, 'hop_length':hop_i})
        real_x_list, y_list = standar_data(real_x_list, y_list, 1, 1, test_few=False)

        img_x_list, y_list = load_data(path_root=PATH_ROOT, form_src='imgspectrum', scaler_data='or', sub_set_way='rand',
                                       **{'win_length':win_i, 'hop_length':hop_i})
        img_x_list, y_list = standar_data(img_x_list, y_list, 1, 1, test_few=False)

        x_list = [np.concatenate([real_i, img_i], axis=-1) for real_i, img_i in zip(real_x_list, img_x_list)]

        logging.debug('X shape')
        for x_i in x_list:
            logging.debug(x_i.shape)

        for i in range(1, 2):
            for j in range(-3, -4, -1):
                search_models(x_list, y_list, [200, 20, 21, 22, 23, 24], path_save, **{'i':i, 'j':j, 'subset_acc_name':subset_acc_name})

        test_all_check_models(path_save, x_list=x_list, y_list=y_list,
                              num_models=[200, 20, 21, 22, 23, 24], model_load=3,
                              dict_model_load={**DICT_MODEL_CONFIG, **DICT_MODEL_STRUCT},
                              kw_model='.hdf5',
                              **DICT_MODEL_COMPILE)

    logging.info('finished')
