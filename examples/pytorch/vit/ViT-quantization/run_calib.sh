python -m torch.distributed.launch --nproc_per_node 1 \
    --master_port 12345 main.py \
    --calib \
    --name vit \
    --pretrained_dir ViT-B_16.npz \
    --data-path data/imagenet \
    --model_type ViT-B_16 \
    --img_size 384 \
    --num-calib-batch 20 \
    --calib-batchsz 2 \
    --quant-mode ft2 \
    --calibrator percentile \
    --percentile 99.99 \
    --batch-size 16 \
    --calib-output-path .

    # original --batch-size 32 \
    # add 'CUDA_VISIBLE_DEVICES'