# Evaluating Memory and Credit Assignment in Memory-Based RL
This is the official code for the paper (Section 5.1 & 5.2: discrete control)

["When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment"](https://arxiv.org/abs/2307.03864), **NeurIPS 2023 (oral)**

by [Tianwei Ni](https://twni2016.github.io/), [Michel Ma](https://scholar.google.com/citations?user=capMFX8AAAAJ&hl=en), [Benjamin Eysenbach](https://ben-eysenbach.github.io/), and [Pierre-Luc Bacon](http://pierrelucbacon.com/). 

Please switch to [the branch](https://github.com/twni2016/Memory-RL/tree/pybullet_jax) to check the code for Section 5.3 (PyBullet continuous control). 

## Modular Design
The code has a modular design which requires *three* configuration files. We hope that such design could facilitate future research on different environments, RL algorithms, and sequence models.

- `config_env`: specify the environment, with `config_env.env_name` specifying the exact (memory / credit assignment) length of the task
    - Passive T-Maze (this work)
    - Active T-Maze (this work)
    - Passive Visual Match (based on [Hung et al., 2018])
    - Key-to-Door (based on [Raposo et al., 2021])
- `config_rl`: specify the RL algorithm and its hyperparameters
    - DQN (with epsilon greedy)
    - SAC-Discrete (we find `--freeze_critic` can prevent gradient explosion, see the discussion in Appendix C.1 in the latest version of the arXiv paper). 
- `config_seq`: specify the sequence model and its hyperparameters including training sequence length `config_seq.sampled_seq_len` and number of layers `--config_seq.model.seq_model_config.n_layer` 
    - LSTM [Hochreiter and Schmidhuber, 1997]
    - Transformer (GPT-2) [Radford et al., 2019]

## Installation
We use python 3.7+ and list the basic requirements in [`requirements.txt`](https://github.com/twni2016/Memory-RL/blob/main/requirements.txt). 

## Reproducing the Results
Below are example commands to reproduce the *main* results shown in Figure 3 and 6. 
For the ablation results, please adjust the corresponding hyperparameters.

To run Passive T-Maze with a memory length of 50 with LSTM-based agent:
```bash
python main.py \
    --config_env configs/envs/tmaze_passive.py \
    --config_env.env_name 50 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 20000 \
    --config_seq configs/seq_models/lstm_default.py \
    --config_seq.sampled_seq_len -1 \
```

To run Passive T-Maze with a memory length of 1500 with Transformer-based agent:
```bash
python main.py \
    --config_env configs/envs/tmaze_passive.py \
    --config_env.env_name 1500 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 6700 \
    --config_seq configs/seq_models/gpt_default.py \
    --config_seq.sampled_seq_len -1 \
```

To run Active T-Maze with a memory length of 20 with Transformer-based agent:
```bash
python main.py \
    --config_env configs/envs/tmaze_active.py \
    --config_env.env_name 20 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 40000 \
    --config_seq configs/seq_models/gpt_default.py \
    --config_seq.sampled_seq_len -1 \
    --config_seq.model.seq_model_config.n_layer 2 \
    --config_seq.model.seq_model_config.n_head 2 \
```

To run Passive Visual Match with a memory length of 60 with Transformer-based agent:
```bash
python main.py \
    --config_env configs/envs/visual_match.py \
    --config_env.env_name 60 \
    --config_rl configs/rl/sacd_default.py \
    --shared_encoder --freeze_critic \
    --train_episodes 40000 \
    --config_seq configs/seq_models/gpt_cnn.py \
    --config_seq.sampled_seq_len -1 \
```

To run Key-to-Door with a memory length of 120 with LSTM-based agent:
```bash
python main.py \
    --config_env configs/envs/keytodoor.py \
    --config_env.env_name 120 \
    --config_rl configs/rl/sacd_default.py \
    --shared_encoder --freeze_critic \
    --train_episodes 40000 \
    --config_seq configs/seq_models/lstm_cnn.py \
    --config_seq.sampled_seq_len -1 \
    --config_seq.model.seq_model_config.n_layer 2 \
```

To run Key-to-Door with a memory length of 250 with Transformer-based agent:
```bash
python main.py \
    --config_env configs/envs/visual_match.py \
    --config_env.env_name 250 \
    --config_rl configs/rl/sacd_default.py \
    --shared_encoder --freeze_critic \
    --train_episodes 30000 \
    --config_seq configs/seq_models/gpt_cnn.py \
    --config_seq.sampled_seq_len -1 \
    --config_seq.model.seq_model_config.n_layer 2 \
    --config_seq.model.seq_model_config.n_head 2 \
```

The `train_episodes` of each task is specified in [`budget.py`](https://github.com/twni2016/Memory-RL/blob/main/budget.py). 

By default, the logging data will be stored in `logs/` folder with csv format. If you use `--debug` flag, it will be stored in `debug/` folder. 

## Logging and Plotting

After the logging data is stored, you can plot the learning curves and aggregation plots (e.g., Figure 3 and 6) using [`vis.ipynb`](https://github.com/twni2016/Memory-RL/blob/main/vis.ipynb) jupyter notebook.

We also provide our logging data used in the paper shared in [google drive](https://drive.google.com/file/d/1bX8lRtm6IYihCmATzgVU7Enq4xuSFAVq/view?usp=sharing) (< 400 MB).

## Acknowledgement

The code is largely based on prior works:
- [POMDP Baselines](https://github.com/twni2016/pomdp-baselines)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Questions
If you have any questions, please raise an issue (preferred) or send an email to Tianwei (tianwei.ni@mila.quebec).
