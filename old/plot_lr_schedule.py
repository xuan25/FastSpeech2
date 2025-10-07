import sys
import os
# Adjust the import path to the project root (if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastspeech2.config import TrainConfig

# TRAIN_CONFIG_PATH = 'config/LibriTTS/train.yaml'
TRAIN_CONFIG_PATH = 'config/LibriTTS/train_prosody_predictor.yaml'

train_config = TrainConfig.load_from_yaml(TRAIN_CONFIG_PATH)

# init a basic NN
def get_lr_scale(current_step, n_warmup_steps, anneal_steps, anneal_rate):
    lr = np.min(
        [
            np.power(current_step, -0.5),
            np.power(n_warmup_steps, -1.5) * current_step,
        ]
    )
    for s in anneal_steps:
        if current_step > s:
            lr = lr * anneal_rate
    return lr

def get_lr_curve(train_config: TrainConfig):
    lrs = []
    for step in range(train_config.step_config.total_step):
        lr = get_lr_scale(
            step,
            train_config.optimizer_config.warm_up_step,
            train_config.optimizer_config.anneal_steps,
            train_config.optimizer_config.anneal_rate
        )
        lrs.append(lr)
    return lrs

import matplotlib.pyplot as plt
import numpy as np

lrs = get_lr_curve(train_config)
plt.figure(figsize=(10, 5))
plt.plot(np.arange(train_config.step_config.total_step), lrs, label='Learning Rate')
plt.title('Learning Rate Schedule')
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig('lr_schedule.png')