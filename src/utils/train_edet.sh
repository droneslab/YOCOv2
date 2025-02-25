#!/bin/bash

cd ../

for DSET in mars moon asteroid; do
    if [ $DSET = moon ]; then
        DS1=moon64
        DS2=moon100
    elif [ $DSET = mars ]; then
        DS1=mars1
        DS2=mars2
    elif [ $DSET = asteroid ]; then
        DS1=asteroid1
        DS2=asteroid3
    fi

    if [ $DSET = moon ]; then
        python train_efficientdet.py /mnt/Space-Vision/data/yoco_journal/$DSET/ $DS1 $DS2
    else
        python train_efficientdet.py /mnt/Space-Vision/data/yoco_journal/$DSET/ $DS1 $DS2
        python train_efficientdet.py /mnt/Space-Vision/data/yoco_journal/$DSET/ $DS2 $DS1
    fi
done