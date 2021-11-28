###############
# UBC-TFSL - 84 miniImageNet Version - EfficientNet
##############
CUDA_VISIBLE_DEVICES=0 python train_moco_ins_84_efficient.py --batch_size 128 --num_workers 2 --softmax --moco --model efficientnet --dataset miniimagenet --gpu=0 --prefix 'Eff_TFSL' --data_folder='../miniimagenettools/ubc-tfsl/'


###############
# UBC-TFSL - 84 miniImageNet Version
##############
CUDA_VISIBLE_DEVICES=0 python train_moco_ins_84.py --batch_size 32 --num_workers 2 --softmax --moco --model resnet12 --dataset miniimagenet --gpu=0 --prefix 'TFSL' --data_folder='../miniimagenettools/ubc-tfsl/'

###############
# 84 miniImageNet Version with resnet 12 code from rfs
##############
CUDA_VISIBLE_DEVICES=0 python train_moco_ins_84.py --batch_size 128 --num_workers 2 --softmax --moco --model resnet12 --dataset miniimagenet --gpu=0

# resume training
CUDA_VISIBLE_DEVICES=0 python train_moco_ins_84.py --batch_size 128 --num_workers 2 --softmax --moco --model resnet12 --dataset miniimagenet --gpu=0 --resume=12_84_miniimagenet_models/84_MoCo0.999_softmax_16384_resnet12_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ/current.pth



###############
# 84 miniImageNet Version
##############
CUDA_VISIBLE_DEVICES=0 python train_moco_ins_84.py --batch_size 128 --num_workers 2 --nce_k 16384 --softmax --moco --model resnet12 --dataset miniimagenet --gpu=0


###############
# older one
##############
CUDA_VISIBLE_DEVICES=0 python eval_moco_ins.py --model resnet12 --model_path \
miniimagenet_models/MoCo0.999_softmax_16384_resnet12_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ/current.pth --num_workers 2 --learning_rate 10 --gpu=0 --batch_size 128

CUDA_VISIBLE_DEVICES=0 python train_moco_ins.py \
 --batch_size 128 --num_workers 2 --nce_k 16384 --softmax --moco --model resnet12 --dataset miniimagenet --gpu=0 --resume=miniimagenet_models/MoCo0.999_softmax_16384_resnet12_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ/current.pth

 python test.py \
 --batch_size 128 --num_workers 2 --nce_k 16384 --softmax --moco --model resnet12 --dataset miniimagenet



