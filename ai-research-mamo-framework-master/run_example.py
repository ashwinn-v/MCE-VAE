"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Template for using the multi-objective trainer.

This script is a template on how to use the multi-objective
trainer to train and develop models. The user is encouraged
to look at the details of this script, since this is the
intended way on how to use this framework. Before running
the script, please set the arguments as shown in the example
on how to run the script. For more information on where to
download the data from, the requirements, etc. please refer
to the README.md of the framework.

Example how to run the script:
    python run_example.py --data_dir bar --models_dir foo

Arguments:
    --data_dir: the path to the directory where the data is stored
    --models_dir: the path to the directory where to save the models,
        it must be empty directory
"""

import torch
import numpy as np
import os
import logging
import sys

from dataloader.ae_data_handler import AEDataHandler
from models.multi_VAE import MultiVAE
from loss.vae_loss import VAELoss
from metric.recall_at_k import RecallAtK
from metric.revenue_at_k import RevenueAtK
from trainer import Trainer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default='/home/blagoj/working_dir/data/',
                    help='the path to the directory where the data is stored')
parser.add_argument('--models_dir',
                    default='/home/blagoj/working_dir/models',
                    help='the path to the directory where to save the models,'
                    + ' it must be empty')
args = parser.parse_args()

# get the arguments
dir_path = args.data_dir
save_to_path = args.models_dir

# set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

# set cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_path = os.path.join(
    dir_path, 'movielens_small_training.npy')
validation_input_data_path = os.path.join(
    dir_path, 'movielens_small_validation_input.npy')
validation_output_data_path = os.path.join(
    dir_path, 'movielens_small_validation_test.npy')
test_input_data_path = os.path.join(
    dir_path, 'movielens_small_test_input.npy')
test_output_data_path = os.path.join(
    dir_path, 'movielens_small_test_test.npy')
products_data_path = os.path.join(
    dir_path, 'movielens_products_data.npy')

data_handler = AEDataHandler(
    'MovieLensSmall', train_data_path, validation_input_data_path,
    validation_output_data_path, test_input_data_path,
    test_output_data_path)

input_dim = data_handler.get_input_dim()
output_dim = data_handler.get_output_dim()

products_data_np = np.load(products_data_path)
products_data_torch = torch.tensor(
    products_data_np, dtype=torch.float32).to(device)

# create model
model = MultiVAE(params='yaml_files/params_multi_VAE_training.yaml')

correctness_loss = VAELoss()
revenue_loss = VAELoss(weighted_vector=products_data_torch)
losses = [correctness_loss, revenue_loss]

recallAtK = RecallAtK(k=10)
revenueAtK = RevenueAtK(k=10, revenue=products_data_np)
validation_metrics = [recallAtK, revenueAtK]

trainer = Trainer(data_handler, model, losses,
                  validation_metrics, save_to_path)
trainer.train()
print(trainer.pareto_manager._pareto_front)
