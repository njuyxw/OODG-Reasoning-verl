### Reproduction Instructions

#### Setup
1. Obtain model and dataset:
    1. Obtain `Qwen/Qwen2.5-7B-Instruct-1M` and place it in:  
    `experiments/models/Qwen2.5-7B-Instruct-1M`
    ```
    bash download.sh
    ```
    2. Download the knights-and-knaves dataset from [HF Datasets](https://huggingface.co/datasets/K-and-K/knights-and-knaves) and place it in:  
    `experiments/raw/knights-and-knaves`
    ```
    cd experiments
    mkdir raw
    cd raw
    git clone https://huggingface.co/datasets/K-and-K/knights-and-knaves 
    ```

2. Create environment:
   ```bash
   conda create -n verl python==3.9
   conda activate verl
   pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
   pip3 install flash-attn --no-build-isolation
   cd verl && pip3 install -e .
   ```

#### Preprocessing
```bash
cd experiments
python ../verl/examples/data_preprocess/kk.py \
  --local_dir ./dataset/kk/instruct/5ppl \
  --data_path ./raw/knights-and-knaves/train/people5_num1000.jsonl
```

#### Training
**Phase 1 (For 100 steps):**  
*Parameters: rollout.n=8, rollout.temperature=1.0*
```bash
bash run_logicRL_4gpus_phase1.sh
```

**Phase 2 (Additional 280 steps):**  
*Parameters updated: rollout.n=16, rollout.temperature=1.3*
```bash
bash run_logicRL_4gpus_phase2.sh
```

You can modify the script to train additional steps on more data to reach better performance.

### Evaluation
```bash
python python ../verl/scripts/model_merger.py --local_dir ./checkpoints/logic_rl/grpo_run/global_step_380/actor/

bash ../evaluation/kk/scripts/eval/eval_grpo.sh
```
