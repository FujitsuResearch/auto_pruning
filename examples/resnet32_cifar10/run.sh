# COPYRIGHT Fujitsu Limited 2023
CUDA_VISIBLE_DEVICES='0' python3 main.py --data ../data --pretrained_model_path ../pretrained_model/pretrained_cifar10_resnet32.pt --use_gpu --use_DataParallel > log.log
