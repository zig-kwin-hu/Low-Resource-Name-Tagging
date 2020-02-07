# Low-Resource-Name-Tagging

This repository shows the implementation of the system described in the paper 
[Low-Resource Name Tagging Learned with Weakly Labeled Data](https://www.aclweb.org/anthology/D19-1025.pdf).

## Environment

python = 2.7

torch = 0.4.1


## Dataset

We collect weakly labeled data from wiki in Mongolian (mn). We select sentences with highest quality as validation set and test set.

## Directory

files/ : input files, you can unzip it from files.zip.

    train.txt local_train.txt nofuzzy_train.txt valid.txt test.txt : examples for training, validation and test.

    entity_dict word_idf_dict mnalphabet : pre-generated files for training.

    word_embedding : word embedding file.

code/ : implementation fo our model.

log/ : log files of traing and evaluation.

## Training and Evaluation

cd code/
python main.py

We output metrics including precision, precision type, recall, recall type, f1 and f1 type. The difference between xx and xx type is that the former one only considers prediction's boundary while that latter one considers both boundary and type.




