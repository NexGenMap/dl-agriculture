### Notes

* For mapping center pivot irrigation systems, please, see our latest work [here](https://github.com/mapbiomas-brazil/irrigation/tree/master/center-pivot-irrigation-systems).

### Get gdal development libraries:

```
$ sudo apt-add-repository ppa:ubuntugis/ubuntugis-unstable
$ sudo apt-get update
$ sudo apt-get install libgdal-dev
$ sudo apt-get install python3-dev
$ sudo apt-get install gdal-bin python3-gdal
```

### Create and activate a virtual environment:

```
$ virtualenv env -p python3
$ source env/bin/activate
```

### Install GDAL:

```
(env) $ pip3 install numpy
(env) $ pip3 install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
```

### Install TensorFlow-GPU
```
(env) $ pip3 install tensorflow-gpu
```

### Install Others Requirements

```
(env) $ pip3 install -r requirements.txt
```


### Build datasets
#### Build dataset to train
```
python3 run.py --mode=generate --image=image1.tif --labels=image1_labels.tif --output=train.h5 --chip_size=512 --channels=4 --grids=2 --rotate=true
python3 run.py --mode=generate --image=image2.tif --labels=image2_labels.tif --output=train.h5 --chip_size=512 --channels=4 --grids=2 --rotate=true

```

#### Build dataset to test
```
python3 run.py --mode=generate --image=image3.tif --labels=image3_labels.tif --output=test.h5 --chip_size=512 --channels=4 --grids=2 --rotate=true
python3 run.py --mode=generate --image=image4.tif --labels=image4_labels.tif --output=test.h5 --chip_size=512 --channels=4 --grids=2 --rotate=true

```

#### Build dataset to validation
```
python3 run.py --mode=generate --image=image5.tif --labels=image5_labels.tif --output=validation.h5 --chip_size=512 --channels=4 --grids=2 --rotate=true
```

### Train model

```
python3 run.py --mode=train --train=train.h5 --test=test.h5 --epochs=100 --batch_size=5 --classes=2
```

The trained model is available here: https://drive.google.com/file/d/11RO6vJL6eYmtz2YlsEGmz2hUPJ3H1rqd/view?usp=sharing

### Evaluate model

```
python3 run.py --mode=evaluate --evaluate=validation.h5 --batch_size=5 --classes=2
```

### Predict image

```
python3 run.py --mode=predict --input=image.tif --output=output.tif --chip_size=1024 --channels=4 --grids=1 --batch_size=5 --classes=2
```

### Paper
Saraiva, M.; Protas, É.; Salgado, M.; Souza, C. Automatic Mapping of Center Pivot Irrigation Systems from Satellite Images Using Deep Learning. Remote Sens. 2020, 12, 558. DOI: [10.3390/rs12030558](https://doi.org/10.3390/rs12030558)
