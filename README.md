We've prepared a Jupyter notebook that will guide you through the process of building and training a Traffic Sign Classification network with Keras.

- Download the [Jupyter notebook](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580ad790_traffic-sign-classification-with-keras/traffic-sign-classification-with-keras.ipynb).  

You may use the train and test data from Project 2 or download it from here:
- Download the [training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580ad86d_train/train.p).
- Download the [test data](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580ad89e_test/test.p).

[Go to class!](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/475f027b-6cb2-47a6-a15b-0714f7fdef18/concepts/46dc38c4-d3ec-4ac9-a011-91db5bd4c3ad)

Software requirements:

- TensorFlow
- Python 3

Setup

```$ conda create -n keras python=3.5```  
```$ source activate keras```  
```$ conda install -c conda-forge tensorflow```   

*NOTE* If you want to use the GPU version, follow the instructions [here](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#using-pip)

```$ pip install keras```  
```$ pip install h5py```  
```$ pip install scipy```  
```$ pip install scikit-learn```  
```$ pip install Pillow```  
```$ mkdir ~/.keras && echo '{"image_dim_ordering": "tf","epsilon": 1e-07,"backend": "tensorflow","floatx": "float32"}' > ~/.keras/keras.json```  


Run this command in the terminal to get started:

```jupyter notebook traffic-sign-classification-with-keras.ipynb```

This will bring up a browser window with the Jupyter notebook. The notebook will guide you in using Keras to create a deep neural network for traffic sign classification.

This is a self-assessed lab.

###Help

Remember that you can get assistance from your mentor, the forums, or the Slack channel. You can also review the concepts from the previous lessons, or consult external documentation.


**Supporting Materials**


Train
Test

- [Traffic Sign Classification With Keras](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580ad790_traffic-sign-classification-with-keras/traffic-sign-classification-with-keras.ipynb)
- [Train](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580ad86d_train/train.p)
- [Test](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580ad89e_test/test.p)