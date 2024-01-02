# Evaluating Memory-Based RL in the PyBullet benchmark (JAX/Flax)

This is the official code for the paper (Section 5.3: PyBullet continuous control)

["When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment"](https://arxiv.org/abs/2307.03864), **NeurIPS 2023 (oral)**

by [Tianwei Ni](https://twni2016.github.io/), [Michel Ma](https://scholar.google.com/citations?user=capMFX8AAAAJ&hl=en), [Benjamin Eysenbach](https://ben-eysenbach.github.io/), and [Pierre-Luc Bacon](http://pierrelucbacon.com/). 

## Modular Design
The code has a modular design which requires *three* configuration files. We hope that such design could facilitate future research on different environments, RL algorithms, and sequence models.

- `config_env`: specify the environment, with `config_env.env_name` specifying the exact task
    - PyBullet-P tasks `pybullet_p.py`
    - PyBullet-V tasks `pybullet_v.py`
- `config_rl`: specify the RL algorithm and its hyperparameters
    - TD3 `td3_default.py`
    - SAC `sac_default.py`
- `config_seq`: specify the sequence model and its hyperparameters including training sequence length `config_seq.sampled_seq_len`
    - LSTM `lstm_default.py`
    - Transformer `gpt_default.py`

## Reproducing the Results in Figure 8

To run LSTM TD3 (*shared* actor and critic) on PyBullet Cheetah-P with sampled sequence length of 64:
```bash
python main.py \
    --config_env configs/envs/pomdps/pybullet_p.py \
    --config_env.env_name cheetah \
    --config_rl configs/rl/td3_default.py \
    --config_seq configs/seq_models/lstm_default.py \
    --config_seq.sampled_seq_len 64 \
    --train_episodes 1500 \
    --shared_encoder --freeze_all \
```

To run GPT TD3 (*shared* actor and critic) on PyBullet Cheetah-V with sampled sequence length of 64:
```bash
python main.py \
    --config_env configs/envs/pomdps/pybullet_v.py \
    --config_env.env_name cheetah \
    --config_rl configs/rl/td3_default.py \
    --config_seq configs/seq_models/gpt_default.py \
    --config_seq.sampled_seq_len 64 \
    --train_episodes 1500 \
    --shared_encoder --freeze_all \
```

By default, the logging data will be stored in `logs/` folder with csv format. If you use `--debug` flag, it will be stored in `debug/` folder. 

## Logging and Plotting

After the logging data is stored, you can plot the learning curves (Figure 8) using [`vis.ipynb`](https://github.com/twni2016/Memory-RL/blob/pybullet_jax/vis.ipynb) jupyter notebook.

We also provide our logging data used in Figure 8 shared in [google drive](https://drive.google.com/file/d/11dVb-4j7KvS0mrZ-2avl2TwyeaE3jTva/view?usp=sharing) (< 30 MB).


# Acknowledgement

The code is largely based on prior works:

- [jaxrl](https://github.com/ikostrikov/jaxrl) in JAX/Flax
- [walk in the park](https://github.com/ikostrikov/walk_in_the_park) in JAX/Flax
- [pomdp-baselines](https://github.com/twni2016/pomdp-baselines) in PyTorch

