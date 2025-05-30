source ~/.bashrc
source ~/anaconda3/bin/activate vllm2

# environment variables
export OMP_NUM_THREADS=4
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

code_base=/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/src/data_curation/data_format_correction
cd $code_base
job_id=5058165
export SLURM_JOB_ID=${job_id}
# unset SLURM_STEP_ID

gpus=8
cpus=64
quotatype="spot"
OMP_NUM_THREADS=1 srun --partition=MoE --jobid=${job_id} --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python correct_format.py \
--pretrained /mnt/petrelfs/songmingyang/quxiaoye/models/Llama-3.1-70B-Instruct \
--input_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/reasoning/tool_dataset/datasets/chargemma_final/format_validation/opensource_version/chartgemma-combined-sharegpt.json \
--output_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/reasoning/tool_dataset/datasets/chargemma_final/format_validation/opensource_version/chartgemma-combined-sharegpt-llama3.1correct-middle.jsonl 



# --input_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/reasoning/tool_dataset/datasets/chargemma_final/format_validation/chartgemma-reachqa-combined-sharegpt.json \
# --output_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/reasoning/tool_dataset/datasets/chargemma_final/format_validation/chartgemma-reachqa-combined-sharegpt-llama3.1correct-middle.jsonl 


# salloc --partition=MoE --job-name="eval" --gres=gpu:8 -n1 --ntasks-per-node=1 -c 64 --quotatype="reserved"
# salloc --partition=MoE --job-name="interact" --gres=gpu:1 -n1 --ntasks-per-node=1 -c 16 --quotatype="reserved"
