### Get gdal development libraries:

```
$ sudo apt-add-repository ppa:ubuntugis/ubuntugis-unstable
$ sudo apt-get update
$ sudo apt-get install libgdal-dev
$ sudo apt-get install python-dev
$ sudo apt-get install gdal-bin python-gdal python3-gdal
```

### Create and activate a virtual environment:

```
$ virtualenv env
$ source env/bin/activate
```

### Install GDAL:

```
(env) $ pip install numpy
(env) $ pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
```

### Install TensorFlow-GPU
```
(env) $ pip install tensorflow-gpu
```

### Install Others Requirements

```
(env) $ pip install -r requirements.txt
```


### Build datasets
#### Build dataset to train
```
python3 run.py --mode=generate --image=image1.tif --labels=image1_labels.tif --output=train.h5 --chip_size=256 --channels=4 --grids=2 --rotate=true --flip=true
python3 run.py --mode=generate --image=image2.tif --labels=image2_labels.tif --output=train.h5 --chip_size=256 --channels=4 --grids=2 --rotate=true --flip=true

```

#### Build dataset to test
```
python3 run.py --mode=generate --image=image3.tif --labels=image3_labels.tif --output=test.h5 --chip_size=256 --channels=4 --grids=2 --rotate=true --flip=true
python3 run.py --mode=generate --image=image4.tif --labels=image4_labels.tif --output=test.h5 --chip_size=256 --channels=4 --grids=2 --rotate=true --flip=true

```

#### Build dataset to validation
```
python3 run.py --mode=generate --image=image5.tif --labels=image5_labels.tif --output=validation.h5 --chip_size=256 --channels=4 --grids=2 --rotate=true --flip=true
```

### Train model

```
python3 run.py --mode=train --train=train.h5 --test=test.h5 --epochs=100 --batch_size=25
```

### Evaluate model

```
python3 run.py --mode=evaluate --evaluate=validation.h5 --batch_size=25
```

### Predict image

```
python3 run.py --mode=predict --input=image.tif --output=output.tif --chip_size=1024 --channels=4 --grids=1 --batch_size 5
```
