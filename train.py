import os, sys
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.append(code_dir)

from omegaconf import OmegaConf
import argparse
from utils.build_utils import build_checkpoint_logger, build_trainer_evaluator, build_data_loaders
from utils.random import set_seed
from models.build_model import build_model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.config
    config = OmegaConf.load(config_path)
    
    checkpoint, logger = build_checkpoint_logger(config)
    config = checkpoint.config
    print(config)

    data_loaders = build_data_loaders(config, config.seed)

    set_seed(config.seed)

    model = build_model(config.model)
    pretrain_checkpoint = config.get('pretrain_checkpoint', None)
    if pretrain_checkpoint is not None:
        pretrain_checkpoint, _ = build_checkpoint_logger(OmegaConf.create({'exp_name':pretrain_checkpoint}), 
                                                         need_logger=False)
        model.load_state_dict(pretrain_checkpoint.best_val_param)

    trainer, val_evaluator, test_evaluators = build_trainer_evaluator(config, model, data_loaders)
    checkpoint.set_trainer(trainer)

    num_epochs = trainer.num_epochs
    start_epoch = checkpoint.start_epoch

    print('Start training.')
    for epoch in range(start_epoch, num_epochs+1):

        # train
        trainer.train()

        epoch_info = f'[Epoch {epoch}/{num_epochs}]'

        # val
        val_metrics, val_info = val_evaluator.evaluate()
        logger.info(epoch_info + ' Validation ' + val_info)
        checkpoint.step(val_metrics['loss'], save_current_weights=True)

    # test with best val model
    model.load_state_dict(checkpoint.best_val_param)
    for key, test_evaluator in test_evaluators.items():
        test_metrics, test_info = test_evaluator.evaluate()
        logger.info(f'{key} ' + test_info)

if __name__ == '__main__':
    main()

