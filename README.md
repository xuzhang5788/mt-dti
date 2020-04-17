# MT-DTI
An official Molecule Transformer Drug Target Interaction (MT-DTI) model

* **Author**: [Bonggun Shin](mailto:bonggun.shin@deargen.me)
* **Paper**: Shin, B., Park, S., Kang, K. & Ho, J.C.. (2019). [Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction](http://proceedings.mlr.press/v106/shin19a/shin19a.pdf). Proceedings of the 4th Machine Learning for Healthcare Conference, in PMLR 106:230-248

## Required Files

* Download [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing)
	* This includes;
		* Orginal KIBA dataset from [DeepDTA](https://github.com/hkmztrk/DeepDTA)
		* tfrecord for KIBA dataset
		* Pretrained weights of the molecule transformer
		* Finetuned weights of the MT-DTI model for KIBA fold0
* Unzip it (folder name is **data**) and place under the project root

```
cd mtdti_demo
# place the downloaded file (data.tar.gz) at "mtdti_demo"
tar xzfv data.tar.gz
```

* These files sholud be in the right places

```
mtdti_demo/data/chembl_to_cids.txt
mtdti_demo/data/CID_CHEMBL.tsv
mtdti_demo/data/kiba/*
mtdti_demo/data/kiba/folds/*
mtdti_demo/data/kiba/mbert_cnn_v1_lr0.0001_k12_k12_k12_fold0/*
mtdti_demo/data/kiba/tfrecord/*.tfrecord
mtdti_demo/data/pretrain/*
mtdti_demo/data/pretrain/mbert_6500k/*
```



## VirtualEnv

* install mkvirtualenv
* create a dti env with the following commands

```
mkvirtualenv --python=`which python3` dti
pip install tensorflow-gpu==1.12.0
```


## Preprocessing

* If downloaded [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing), then you can skip these preprocessings


* Transform kiba dataset into one pickle file

```
python kiba_to_pkl.py 

# Resulted files
mtdti_demo/data/kiba/kiba_b.cpkl
```



* Prepare Tensorflow Record files

```
cd src/preprocess
export PYTHONPATH='../../'
python tfrecord_writer.py 

# Resulted files
mtdti_demo/data/kiba/tfrecord/*.tfrecord
```

## FineTuning

* If downloaded [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing), then you can skip this finetuning

```
cd src/finetune
export PYTHONPATH='../../'
python finetune_demo.py 

```


## Prediction

```
cd src/predict
export PYTHONPATH='../../'
python predict_demo.py 
```



