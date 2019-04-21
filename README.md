# 50.039 Project


## Project Description
The Image Captioning project of 50.039

## User Manual
### Set up training environment

1. Install libraries

The following libraries need to be installed:
```
Python: 3.6 above
Pytorch: 1.0.1.post2
nltk: 3.4
cython: 0.29.6
python3-dev: 3.6.7-1~18.04
```

2. Clone API repositories

COCO API is used for our image dataset. To setup COCO, run the following command in Terminal:
```
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
```

3. Modify directory
**Path of files**
```
Training images: “./data2/train2014/*.jpg”
Validation images: “./data2/val2014/*.jpg”
models: “./data2/models/*.ckpt”
Vocabulary: “./data2/vocab.pkl”
```

4. Preprocessing

Before training, we need to finish the following steps for pre-processing:

* build the vocabulary

* resize training images and validation images for training
```
$ python3 build_vocab.py   
$ python3 resize.py
$ python3 resize.py --image_dir ‘./data2/val2014’ --output_dir ‘./data2/val_resized2014’
```

### Training and Validating

After finishing each epoch, there will be a validation for the whole validation set.

The configuration for training can be altered using the parser.
```
$ python3 train.py --num_epochs 10
```

### Test the model

Run the demonstrator to test the performance of models on single image prediction. Make sure the following files are under the same directory:
```
demonstrator.py
demonstrator.kv
sample.py
model.py
vocab.pkl
```
Run the following command in Terminal:
```
$ python3 demonstrator.py
```
The models used for the sampling are our final models in default: ```/data2/models/encoder.ckpt``` and ```/data2/models/decoder.ckpt```

To test with other encoders and decoders, kindly modify the default model path (Windows) or change the model path in the parser (UNIX system).

**How to use the demonstrator:**

Go into folders to select an image on the left and press the button “Predict on image”, then the image and prediction will be displayed on the right side of the interface.
