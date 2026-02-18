# Reproducing TiTok-B-64
torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:9999 sample_imagenet_titok.py \
config=configs/infer_optvq_b64.yaml \
experiment.output_dir="titok_b_64" \
experiment.generator_checkpoint="outputs/optvq_b64_maskgit_shared_run3/checkpoint-100000/ema_model/pytorch_model.bin"

# Run eval script. The result FID should be ~2.48
# python3 guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz titok_b_64.npz
