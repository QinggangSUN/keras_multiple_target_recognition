#  keras_multi_target_signal_recognition
Underwater single channel acoustic multiple targets recognition using
ResNet, DenseNet, and Complex-Valued convolutional nerual networks.
keras-gpu 2.2.4 with tensorflow-gpu 1.12.0 backend.

#  How to use
Experiment one:
    Eight class recognition.
    run prepare_data_shipsear_recognition_mix_s0tos3.py
        recognition_mix_shipsear_s0tos3_preprocess.py
        to preprocess datas.
    run train_recognition_mix.py
        to train models.
Experiment two:
    Twenty class recognition.
    run prepare_data_shipsear_recognition_mix_s0tos3full3.py
        recognition_mix_shipsear_s0tos3full3_preprocess.py
        to preprocess datas.
    run train_recognition_mix_full3.py
        to train models.

#  Reference
Please cite the original work as:  

[Deep Complex Networks](https://github.com/ChihebTrabelsi/deep_complex_networks)  
@ARTICLE {,   
author  = "Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal",  
title   = "Deep Complex Networks",  
journal = "arXiv preprint arXiv:1705.09792",  
year    = "2017"
}  
  
[Complex-Valued Neural Networks in Keras with Tensorflow](https://github.com/zengjie617789/keras-complex)  
@misc{dramsch2019complex,  
title     = {Complex-Valued Neural Networks in Keras with Tensorflow},  
url       = {https://figshare.com/articles/Complex-Valued_Neural_Networks_in_Keras_with_Tensorflow/9783773/1},   
DOI       = {10.6084/m9.figshare.9783773},  
publisher = {figshare},   
author    = {Dramsch, Jesper S{\"o}ren and Contributors}, 
 year      = {2019}
}

[ResNet](https://arxiv.org/abs/1512.03385)

[Keras-ResNet](https://github.com/broadinstitute/keras-resnet)

[DenseNet](https://arxiv.org/pdf/1608.06993v3.pdf)

[Dense Net in Keras](https://github.com/titu1994/DenseNet)

[densenet_1d](https://github.com/ankitvgupta/densenet_1d)

#  Related works
[Deep Complex Networks (ICLR 2018)](https://arxiv.org/abs/1705.09792)  
[On Complex Valued Convolutional Neural Networks](https://arxiv.org/abs/1602.09046)  
[Spectral Representations for Convolutional Neural Networks](https://arxiv.org/abs/1506.03767)  
[Complex-Valued_Networks](https://github.com/wangyifan1027/Complex-Valued_Networks)  
[Deep Complex Networks](https://github.com/Doyosae/Deep_Complex_Networks)  
[Complex-Valued MRI Reconstruction - Unrolled Architecture](https://github.com/MRSRL/complex-networks-release)  
[Complex ResNet Aided DoA Estimation for Near-Field MIMO Systems](https://arxiv.org/abs/2007.10590)  
[Complex-Valued Densely Connected Convolutional Networks](https://doi.org/10.1007/978-981-15-7981-3_21)
