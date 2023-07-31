# Evaluating Memory and Credit Assignment in Memory-Based RL
This is the official code for the paper ["When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment"](https://arxiv.org/abs/2307.03864).  

## Modular Design
The code has a modular design which requires *three* configuration files. We hope that such design could facilitate future research on different environments, RL algorithms, and sequence models.

- `config_env`: specify the environment, with `config_env.env_name` specifying the exact (memory / credit assignment) length of the task
    - T-Maze Passive
    - T-Maze Active
    - Passive Visual Match
    - Key-to-Door
- `config_rl`: specify the RL algorithm and its hyperparameters
    - DQN (with epsilon greedy)
    - SAC-Discrete (we find `--freeze_critic` can prevent gradient explosion found in [prior work](https://arxiv.org/abs/2110.05038))
- `config_seq`: specify the sequence model and its hyperparameters including training sequence length `config_seq.sampled_seq_len` and number of layers `--config_seq.model.seq_model_config.n_layer` 
    - LSTM
    - Transformer (GPT-2)

## Installation
We use python 3.7+ and list the basic requirements in `requirements.txt`. 

## Reproducing the Results
Below are example commands to reproduce the *main* results shown in Figure 3 and 6. 
For the ablation results, please adjust the corresponding hyperparameters.

To run T-Maze Passive with a memory length of 50 with LSTM-based agent:
```bash
python main.py \
    --config_env configs/envs/tmaze_passive.py \
    --config_env.env_name 50 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 20000 \
    --config_seq configs/seq_models/lstm_default.py \
    --config_seq.sampled_seq_len -1 \
```

To run T-Maze Passive with a memory length of 1500 with Transformer-based agent:
```bash
python main.py \
    --config_env configs/envs/tmaze_passive.py \
    --config_env.env_name 1500 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 8000 \
    --config_seq configs/seq_models/gpt_default.py \
    --config_seq.sampled_seq_len -1 \
```

To run T-Maze Active with a memory length of 20 with Transformer-based agent:
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

## Plotting
By default, the logging data will be stored in `logs/` folder with csv format. If you use `--debug` flag, it will be stored in `debug/` folder. 

You can plot the learning curves and aggregation plots (e.g., Figure 3 and 6) using `vis.ipynb` jupyter notebook.

## Acknowledgement

The code is largely based on prior work:
- [POMDP Baselines](https://github.com/twni2016/pomdp-baselines)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Questions
If you have any questions, please raise an issue or send an email to Tianwei (tianwei.ni@mila.quebec).
