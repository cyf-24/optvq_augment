# generator
WANDB_MODE=offline PYTHONPATH=./ HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_maskgit.py config=configs/optvq_maskgit.yaml \
    experiment.project="optvq_generation" \
    experiment.name="optvq_b64_maskgit_AR_run5_new_fram" \
    experiment.output_dir="outputs/optvq_b64_maskgit_AR_run5_new_frame" \
    experiment.tokenizer_checkpoint=outputs/optvq_b64_stage2_shared-dinodisc/checkpoint-1000000/ema_model/pytorch_model.bin