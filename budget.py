"""
This file specifies the `train_episodes` for each task 
    to reproduce our results.
Each dict has (key, value) pairs of `env_name` length and `train_episodes`, 
    and also commented with total env steps.
Also check vis.ipynb for plotting the learning curves and aggregation plots.
"""
tmaze_passive = {
    "50": 20000,  # 1e6 steps
    "100": 20000,  # 2e6
    "250": 16000,  # 4e6
    "500": 8000,  # 4e6
    "750": 8000,  # 6e6
    "1000": 8000,  # 8e6
    "1250": 8000,  # 1e7
    "1500": 6700,  # 1e7
}

tmaze_active = {
    "20": 40000,  # 0.8e6
    "50": 40000,  # 2e6
    "100": 40000,  # 4e6
    "250": 32000,  # 8e6
    "500": 16000,  # 8e6
}

passive_visual_math = {
    "60": 40000,
    "120": 30000,
    "250": 15000,
    "500": 10000,
    "750": 10000,
    "1000": 10000,
}

key_to_door = {
    "60": 40000,
    "120": 40000,
    "250": 30000,
    "500": 15000,
}
