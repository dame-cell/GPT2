# GPT2-small
GPT-small a 92 million parameter Language-model   

<p align="center">
  <img src="images/small_ronot.png" alt="dogs" width="400"/>
</p>

# Examples 


# Training Details 

The model was trained using **two NVIDIA T4 GPUs** in a distributed data parallel (DDP) setup, which significantly sped up the training process. We utilized PyTorch’s `DistributedDataParallel` (DDP) to ensure that the model’s parameters were synchronized across both GPUs during training. 

## Key Training Details

| **Parameter**             | **Value**                               |
|---------------------------|-----------------------------------------|
| GPUs                      | 2x NVIDIA T4                            |
| Batch Size                | 16 per GPU                              |
| Optimizer                 | AdamW (LR: 1e-4, Weight Decay: 1e-4)    |
| Learning Rate Scheduler    | Linear warm-up (10% of total steps)     |
| Mixed Precision            | torch.float16          |
| Epochs                    | 20                                      |
| Evaluation Interval        | Every 2000 step                        |


Model checkpoints are saved every 4 epochs, and validation is performed at regular intervals to monitor performance. 

## Dataset Information

| **Description**            | **Count**       |
|----------------------------|-----------------|
| Total tokens               | 206,130,153      |
| Training tokens            | 185,517,137       |
| Testing tokens             | 20,613,016        |
| Number of training sequences| 724,675    |
| Number of testing sequences | 80,518        |


# Get Started
- First git clone the model and install the requirements 
```bash
git clone https://github.com/dame-cell/GPT2.git
cd GPT2 
pip install -r requirements.txt
cd model 
```
- Then we prepare the dataset
```bash
python3 data.pt --sample_size 100_000 # max is 700k 
```
- After the data is prepared now go to kaggle choose the two t4 gpus options
```bash
python test_ddp2.py --batch_size 16 --eval_batch_size 16 --eval_interval 2000 --epochs 12 --train_data "path to the train_npz" --test_data "path to the test_npz"
```
# Obvervations and Experiments 
