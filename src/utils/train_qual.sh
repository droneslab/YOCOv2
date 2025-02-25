#!/bin/bash
set -e
cd ../

# Mars (roi disc, roi fm disc, roi tk fm disc)
python train.py --arch v5m --epochs 50 --batch_size 32 --train_ds ../datasets/vm/mars1.yaml --test_ds ../datasets/vm/mars_hirise.yaml --qual --yoco --da_type roi --da_loss disc --exp ROI_Disc_50
python train.py --arch v5m --epochs 50 --batch_size 32 --train_ds ../datasets/vm/mars1.yaml --test_ds ../datasets/vm/mars_hirise.yaml --qual --yoco --da_type roi --da_loss disc --fm --exp ROI_FM_Disc_50
python train.py --arch v5m --epochs 50 --batch_size 32 --train_ds ../datasets/vm/mars1.yaml --test_ds ../datasets/vm/mars_hirise.yaml --qual --yoco --da_type tk  --da_loss disc --fm --exp ROI_TK_FM_Disc_50

# Asteroid (YOCOv0, KMeans, PTAP)
# python train.py --arch v6s --epochs 150 --batch_size 32 --train_ds ../datasets/vm/asteroid1.yaml --test_ds ../datasets/vm/orex.yaml --qual --exp Stock_200 --pretrained '/mnt/nas/data/yoco_journal/logs/stocks_midtarget/yolov6s_asteroid1->asteroid3/weights/last.pt'
# python train.py --arch v6s --epochs 150 --batch_size 32 --train_ds ../datasets/vm/asteroid1.yaml --test_ds ../datasets/vm/orex.yaml --qual --yoco --da_type yocov0  --da_loss disc --exp YOCOv0_200 --pretrained '../logs/yocov6s_asteroid1->orex_YOCOv0_50/weights/last.pt'
# python train.py --arch v6s --epochs 150 --batch_size 32 --train_ds ../datasets/vm/asteroid1.yaml --test_ds ../datasets/vm/orex.yaml --qual --yoco --da_type ptap    --da_loss disc --fm --exp PTAP_200 --pretrained '../logs/yocov6s_asteroid1->orex_PTAP_50/weights/last.pt'
# python train.py --arch v6s --epochs 150 --batch_size 32 --train_ds ../datasets/vm/asteroid1.yaml --test_ds ../datasets/vm/orex.yaml --qual --yoco --da_type kmeans  --da_loss disc --fm --exp KMeans_200 --pretrained '../logs/yocov6s_asteroid1->orex_KMeans_50/weights/last.pt'
w