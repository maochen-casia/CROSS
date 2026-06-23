import os, sys
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.append(code_dir)

from omegaconf import OmegaConf
import argparse
from utils.build_utils import build_checkpoint_logger, build_trainer_evaluator, build_data_loaders
from utils.random import set_seed
from models.build_model import build_model

def update_eval_steps(config):

    # always use 33 steps for xy search with max_test_init_offset=20 m
    eval_xy_search_steps = 33

    # orientation search steps
    batch_size = config.data.batch_size
    if config.data.max_test_init_yaw_deg == 0:
        eval_yaw_search_steps = 1
    elif config.data.max_test_init_yaw_deg == 10:
        eval_yaw_search_steps = 17
    elif config.data.max_test_init_yaw_deg == 180:
        eval_yaw_search_steps = 129
        batch_size = 6
    else:
        eval_yaw_search_steps = 33
    
    config.model.eval_xy_search_steps = eval_xy_search_steps
    config.model.eval_yaw_search_steps = eval_yaw_search_steps
    config.data.batch_size = batch_size
    return config

def get_test_info(config):
    max_test_init_offset = config.data.max_test_init_offset
    max_test_init_yaw_deg = config.data.max_test_init_yaw_deg
    eval_xy_search_steps = config.model.eval_xy_search_steps
    eval_yaw_search_steps = config.model.eval_yaw_search_steps
    test_info = f'initial noise: ({max_test_init_offset} m, {max_test_init_yaw_deg} deg), ' \
                f'eval search steps: ({eval_xy_search_steps} xy, {eval_yaw_search_steps} yaw)'
    return test_info

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--max_test_init_yaw_deg', type=float, required=True, help='Override max_test_init_yaw_deg in config.')
    args = parser.parse_args()

    config_path = args.config
    config = OmegaConf.load(config_path)
    
    checkpoint, logger = build_checkpoint_logger(config)
    config = checkpoint.config
    config.data.max_test_init_yaw_deg = args.max_test_init_yaw_deg
    config = update_eval_steps(config)
    print(config)

    data_loaders = build_data_loaders(config, config.seed)

    set_seed(config.seed)

    model = build_model(config.model)
    
    _, val_evaluator, test_evaluator = build_trainer_evaluator(config, model, data_loaders)

    # test with best val model
    model.load_state_dict(checkpoint.best_val_param)

    print('Start Testing')

    logger.info(get_test_info(config))

    val_metrics, val_info = val_evaluator.evaluate()
    logger.info('Validation ' + val_info)

    test_metrics, test_info = test_evaluator.evaluate()
    logger.info('Test ' + test_info)

if __name__ == '__main__':
    main()

