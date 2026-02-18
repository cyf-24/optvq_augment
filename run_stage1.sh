# stage 1
WANDB_MODE=offline PYTHONPATH=./ HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_titok.py config=configs/optvq_b64_stage1.yaml \
    experiment.project=optvq_b64_stage1 \
    experiment.name=optvq_b64_stage1_run2 \
    experiment.output_dir=outputs/optvq_b64_stage1_run2