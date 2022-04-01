#  keras_multi_target_signal_recognition
Underwater single channel acoustic multiple targets recognition using
ResNet, DenseNet, and Complex-Valued convolutional nerual networks.
keras-gpu 2.2.4 with tensorflow-gpu 1.12.0 backend.
#  How to cite this work
This is the official code of the blow article.
Please cite this work in your publications as :  
<pre>
@article{sun_underwater_2022,
title = {Underwater single-channel acoustic signal multitarget recognition using convolutional neural networks},
volume = {151},
issn = {0001-4966},
url = {https://asa.scitation.org/doi/10.1121/10.0009852},
doi = {10.1121/10.0009852},
abstract = {The radiated noise from ships is of great significance to target recognition, and several deep learning methods have been developed for the recognition of underwater acoustic signals. Previous studies have focused on single-target recognition, with relatively few reports on multitarget recognition. This paper proposes a deep learning-based single-channel multitarget underwater acoustic signal recognition method for an unknown number of targets in the specified category. The proposed method allows the two subproblems of recognizing the unique class and duplicate categories of multiple targets to be solved. These two tasks are essentially multilabel binary classification and multilabel multiple value classification, respectively. In this paper, we describe the use of real-valued and complex-valued ResNet and DenseNet convolutional networks to recognize synthetic mixed multitarget signals, which was superimposed from individual target signals. We compare the performance of various features, including the original audio signal, complex-valued short-time Fourier transform (STFT) spectrum, magnitude STFT spectrum, logarithmic mel spectrum, and mel frequency cepstral coefficients. The experimental results show that our method can effectively recognize synthetic multitarget ship signals when the magnitude STFT spectrum, complex-valued STFT spectrum, and log-mel spectrum are used as network inputs.},
language = {en},
number = {3},
journal = {The Journal of the Acoustical Society of America},
author = {Sun, Qinggang and Wang, Kejun},
month = mar,
year = {2022},
pages = {2245--2254},
}  
</pre>
#  How to use
<pre>
download and organize data files as data_dir_tree.md  
install modified package ./models/resnet_broadinstitute  and other dependent packages in the requirements file 
Experiment one:  
    Eight class recognition.
    run
        prepare_data_shipsear_recognition_mix_s0tos3.py
        recognition_mix_shipsear_s0tos3_preprocess.py
    to preprocess datas.
    run
        train_recognition_mix.py
    to train models.
Experiment two:
    Twenty class recognition.
    run
        prepare_data_shipsear_recognition_mix_s0tos3full3.py
        recognition_mix_shipsear_s0tos3full3_preprocess.py
    to preprocess datas.
    run
        train_recognition_mix_full3.py
    to train models.
</pre>
#  Reference
Please cite the original work as :  
[Deep Complex Networks](https://github.com/ChihebTrabelsi/deep_complex_networks)  
<pre>
@ARTICLE {,   
author = "Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal",
title = "Deep Complex Networks",
journal = "arXiv preprint arXiv:1705.09792",
year = "2017"
}
</pre>
[Complex-Valued Neural Networks in Keras with Tensorflow](https://github.com/zengjie617789/keras-complex)  
<pre>
@misc{dramsch2019complex,
title = {Complex-Valued Neural Networks in Keras with Tensorflow},
url = {https://figshare.com/articles/Complex-Valued_Neural_Networks_in_Keras_with_Tensorflow/9783773/1},
DOI = {10.6084/m9.figshare.9783773},
publisher = {figshare},
author = {Dramsch, Jesper S{\"o}ren and Contributors},
year = {2019}
}
</pre>
[ResNet](https://arxiv.org/abs/1512.03385)

[Keras-ResNet](https://github.com/broadinstitute/keras-resnet)

[DenseNet](https://arxiv.org/pdf/1608.06993v3.pdf)

[Dense Net in Keras](https://github.com/titu1994/DenseNet)

[densenet_1d](https://github.com/ankitvgupta/densenet_1d)

[Demystifying Convolutional Neural Networks using GradCam](https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-gradcam-554a85dd4e48)

#  Related works
[Deep Complex Networks (ICLR 2018)](https://arxiv.org/abs/1705.09792)  
[On Complex Valued Convolutional Neural Networks](https://arxiv.org/abs/1602.09046)  
[Spectral Representations for Convolutional Neural Networks](https://arxiv.org/abs/1506.03767)  
[Complex-Valued_Networks](https://github.com/wangyifan1027/Complex-Valued_Networks)  
[Deep Complex Networks](https://github.com/Doyosae/Deep_Complex_Networks)  
[Complex-Valued MRI Reconstruction - Unrolled Architecture](https://github.com/MRSRL/complex-networks-release)  
[Complex ResNet Aided DoA Estimation for Near-Field MIMO Systems](https://arxiv.org/abs/2007.10590)  
[Complex-Valued Densely Connected Convolutional Networks](https://doi.org/10.1007/978-981-15-7981-3_21)
