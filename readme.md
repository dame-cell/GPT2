# GPT2-small
GPT-small a 92 million parameter Language-model   

<p align="center">
  <img src="images/small_ronot.png" alt="cutegpt" width="400"/>
</p>

# Examples 


# Training Details 

The model was trained using **two NVIDIA T4 GPUs** in a distributed data parallel (DDP) setup, which significantly sped up the training process. We utilized PyTorch’s `DistributedDataParallel` (DDP) to ensure that the model’s parameters were synchronized across both GPUs during training. 

## Key Training Details

| **Parameter**             | **Value**                               |
|---------------------------|-----------------------------------------|
| GPUs                      | 2x NVIDIA T4                            |
| Batch Size                | 16 per GPU                              |
| Optimizer                 | AdamW (LR: 6e-4, Weight Decay: 1e-4)    |
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
python3 data.py --sample_size 100_000 # max is 700k 
```
- After the data is prepared now go to kaggle choose the two t4 gpus options
```bash
python test_ddp2.py --batch_size 16 --eval_batch_size 16 --eval_interval 2000 --epochs 12 --train_data "path to the train_npz" --test_data "path to the test_npz"
```
### Observations and Experiments


I performed a few experiments here are some obersavation i discovered :

### Learning Rate
- Started training with a learning rate of `1e-4`, which led to limited progress. The model generated coherent text but lacked contextual relevance, often producing random outputs.

- Increased the learning rate to `3e-4`, leading to faster convergence and more relevant text generation. However, some overfitting occurred.

- To mitigate overfitting, increased both dataset size and learning rate to `6e-4`, resulting in better generalization, contextually relevant text, and faster convergence without overfitting.

### Model Size

- In the first training, the model had over `200 million` parameters. This was not ideal given the small dataset, and training it led to poor results.

- In the second iteration, I reduced the model size to `90+ million` parameters. This time, the model performed significantly better.

These experiments were done after a lot of iteration for instance I had to train the model again and again to make sure it actually generate coherent texts.

If you want to train you own model , you can easily run this code but keep in mind if you run it on only 100k rows for 10 epochs and you will notice it does generate coherent texts 

# Plots 

| Train los | Val loss |
|:-----:|:-------:|
| <img src="images/W&B Chart 9_23_2024, 5_22_28 PM.png" width="500" alt="Tokenization Comparison Hindi"> | <img src="images/W&B Chart 9_23_2024, 5_22_39 PM.png" width="500" alt="Tokenization Comparison English"> | 

<p align="center">
  <img src="images/W&B Chart 9_23_2024, 5_22_46 PM.png" alt="lr" width="500"/>
</p>

# Future Works 
I plan on continually training this model until I'm satisfied with its generation and as i kept training the train loss and the  val loss kept decreasing which  indicate continuous improvement. 


Thanks for reading!! : ) 