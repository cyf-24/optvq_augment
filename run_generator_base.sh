# generator
WANDB_MODE=offline PYTHONPATH=./ HF_ENDPOINT=https://hf-mirror.com /home/zbr/.conda/envs/optvq/bin/accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_maskgit.py config=configs/titok_maskgit.yaml \
    experiment.project="base_generation" \
    experiment.name="base_b64_maskgit_run1" \
    experiment.output_dir="outputs/base_b64_maskgit_run1" \
    experiment.tokenizer_checkpoint="huggingface:yucornetto/tokenizer_titok_b64_imagenet"
