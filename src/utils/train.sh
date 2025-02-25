#!/bin/bash

cd ../

# EPOCHS=200
BSIZE=32

# python train.py --arch v8s --epochs 400 --batch_size 32 --train_ds ../datasets/vm/mars1.yaml --test_ds ../datasets/vm/mars_edl.yaml --qual --exp 400
# python train.py --arch v8s --epochs 400 --batch_size 32 --train_ds ../datasets/vm/mars2.yaml --test_ds ../datasets/vm/mars_edl.yaml --qual --exp 400
# python train.py --arch v8s --epochs 400 --batch_size 32 --train_ds ../datasets/vm/mars1.yaml --test_ds ../datasets/vm/mars_edl.yaml --yoco --fm --da_type tk --da_loss contrastive --exp ROI_TK_FM_Contrastive_400 --qual

# python train.py --arch v8s --epochs 200 --batch_size 32 --train_ds ../datasets/vm/mars1.yaml --test_ds ../datasets/vm/mars_edl.yaml --yoco --da_type roi --da_loss disc --exp ROI_Disc --qual --resume
# python train.py --arch v8s --epochs 200 --batch_size 32 --train_ds ../datasets/vm/mars1.yaml --test_ds ../datasets/vm/mars_edl.yaml --yoco --fm --da_type tk --da_loss disc --exp ROI_TK_FM_Disc --qual --resume
# python train.py --arch v8s --epochs 200 --batch_size 32 --train_ds ../datasets/vm/mars1.yaml --test_ds ../datasets/vm/mars_edl.yaml --yoco --fm --da_type kmeans --da_loss disc --exp KMeans_FM_Disc --qual --resume
# python train.py --arch v8s --epochs 400 --batch_size 32 --train_ds ../datasets/vm/mars1.yaml --test_ds ../datasets/vm/mars_edl.yaml --yoco --da_type yocov0 --da_loss disc --exp YOCOv0 --qual --resume

python train.py --arch v5m --epochs 200 --batch_size 32 --train_ds ../datasets/vm/asteroid1.yaml --test_ds ../datasets/vm/orex.yaml --yoco --fm --da_type tk --da_loss contrastive --exp ROI_TK_FM_Contrastive_200 --qual
python train.py --arch v5m --epochs 200 --batch_size 32 --train_ds ../datasets/vm/asteroid1.yaml --test_ds ../datasets/vm/orex.yaml --yoco --fm --da_type ptap --da_loss disc --exp PTAP_FM_Disc_200 --qual
python train.py --arch v5m --epochs 200 --batch_size 32 --train_ds ../datasets/vm/asteroid1.yaml --test_ds ../datasets/vm/orex.yaml --yoco --da_type roi --da_loss contrastive --exp ROI_Contrastive_200 --qual
# python train.py --arch v5m --epochs 200 --batch_size 32 --train_ds ../datasets/vm/asteroid1.yaml --test_ds ../datasets/vm/orex.yaml --yoco --da_type yocov0 --da_loss disc --exp YOCOv0 --qual


# for ARCH in v5n v5s v5m v6n v6s v8n v8s v8m; do 
# 	for DSET in mars asteroid moon; do
#     		if [ $DSET = moon ]; then
#         		DS1=../datasets/vm/moon64.yaml
#         		DS2=../datasets/vm/moon100.yaml
#     		elif [ $DSET = mars ]; then
#         		DS1=../datasets/vm/mars1.yaml
#         		DS2=../datasets/vm/mars2.yaml
#     		elif [ $DSET = asteroid ]; then
#         		DS1=../datasets/vm/asteroid1.yaml
#         		DS2=../datasets/vm/asteroid3.yaml
#     		fi

# 		#python train.py --arch $ARCH --epochs $EPOCHS --batch_size $BSIZE --train_ds $DS1 --test_ds $DS2 --yoco --da_type roi --da_loss contrastive --exp ROI_Contrastive
# 		#python train.py --arch $ARCH --epochs $EPOCHS --batch_size $BSIZE --train_ds $DS1 --test_ds $DS2 --yoco --fm --da_type roi --da_loss disc --exp ROI_FM_Disc
# 		#python train.py --arch $ARCH --epochs $EPOCHS --batch_size $BSIZE --train_ds $DS1 --test_ds $DS2 --yoco --fm --da_type roi --da_loss contrastive --exp ROI_FM_Contrastive
# 		#python train.py --arch $ARCH --epochs $EPOCHS --batch_size $BSIZE --train_ds $DS1 --test_ds $DS2 --yoco --fm --da_type tk --da_loss disc --exp ROI_TK_FM_Disc
# 		#python train.py --arch $ARCH --epochs $EPOCHS --batch_size $BSIZE --train_ds $DS1 --test_ds $DS2 --yoco --fm --da_type tk --da_loss contrastive --exp ROI_TK_FM_Contrastive
	
# 	#python train.py --arch $ARCH --epochs $EPOCHS --batch_size $BSIZE --train_ds $DS1 --test_ds $DS2 --yoco --fm --da_type kmeans --da_loss disc --exp KMeans_FM_Disc
# 		#python train.py --arch $ARCH --epochs $EPOCHS --batch_size $BSIZE --train_ds $DS1 --test_ds $DS2 --yoco --fm --da_type ptap --da_loss disc --exp PTAP_FM_Disc
# 		python train.py --arch $ARCH --epochs $EPOCHS --batch_size $BSIZE --train_ds $DS1 --test_ds $DS2 --yoco --da_type yocov0 --da_loss disc --exp YOCOv0
# 	done
# done

# for ARCH in v5n v5s v5m v6n v6s v8n v8s v8m; do 
# 	for DSET in mars asteroid; do
#     		if [ $DSET = moon ]; then
#         		DS1=../datasets/vm/moon64.yaml
#         		DS2=../datasets/vm/moon100.yaml
#     		elif [ $DSET = mars ]; then
#         		DS1=../datasets/vm/mars1.yaml
#         		DS2=../datasets/vm/mars2.yaml
#     		elif [ $DSET = asteroid ]; then
#         		DS1=../datasets/vm/asteroid1.yaml
#         		DS2=../datasets/vm/asteroid3.yaml
#     		fi

# 		python train.py --arch $ARCH --epochs $EPOCHS --batch_size $BSIZE --train_ds $DS2 --test_ds $DS1 --yoco --exp ROI.1
# 	done
# done
