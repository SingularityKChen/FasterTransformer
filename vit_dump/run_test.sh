python infer_visiontransformer_int8_op.py \
    --model_type=ViT-B_16  \
    --img_size 384 \
    --calibrated_dir ./ViT-quantization/ViT-B_16_calib.pth \
    --data-path ./ViT-quantization/data/imagenet \
    --batch-size=30 \
    --th-path=$WORKSPACE/build/lib/libth_transformer.so \
    --quant-mode ft2 \
    --validate

    # add 'CUDA_VISIBLE_DEVICES'