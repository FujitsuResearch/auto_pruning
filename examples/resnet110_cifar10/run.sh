CUDA_VISIBLE_DEVICES='0' python3 main.py --model_path ../pretrained_model/pretrained_cifar10_resnet110.pt --use_gpu --use_DataParallel --pruned_model_path ./pruned_cifar10_resnet110_00.pt > log.log
