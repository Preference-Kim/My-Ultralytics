import yaml
from pathlib import Path

from .trainer import MyDetectionTrainer

"""
Example:
    ```bash
    python -m topics.custom_dataset_integration.train
    ```
"""


CFG_PATH = Path(__file__).parent / 'cfg'


if __name__ == '__main__':
    
    with open(CFG_PATH / 'train.yaml', 'r') as f:
        cfgs = yaml.safe_load(f)
    
    data_cfg:Path = CFG_PATH / cfgs['data']
    cfgs['data'] = str(data_cfg.resolve())
    
    trainer = MyDetectionTrainer(overrides=cfgs)
    trainer.train()
