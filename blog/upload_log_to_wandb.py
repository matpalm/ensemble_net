import pandas as pd
import json
import wandb

df = pd.read_csv("logs/None/ax_trials.tsv", sep="\t")
for idx, row in df.iterrows():
    params = eval(row['parameters'])
    wandb.init(project='ensemble_net', group='test_group',
               name=f"test_run_{idx}", reinit=True)
    wandb.config.max_conv_size = params['max_conv_size']
    wandb.config.dense_kernel_size = params['dense_kernel_size']
    wandb.config.learning_rate = params['learning_rate']
    wandb.config.batch_size = params['batch_size']
    wandb.log({'validation_loss': row['final_loss']})
    wandb.join()
