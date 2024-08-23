import yaml
from pathlib import Path
from myultralytics.trainer import MyDetectionTrainer

"""
Prerequisites:
    install `myultralytics` package on root
    ```bash
        pip install -e .
    ```

Example:
    ```bash
        export PIN_MEMORY=False # Optional
        python topics/custom_dataset_integration/train.py
    ```
"""


CFG_PATH = Path(__file__).parent / 'cfg'


def freeze_layer(trainer):
    """
    NOTE:

    This callback doesn't support at DDP.
    If a fine training with multi-GPUs is needed, add `freeze` option in `train.yaml`
    """
    
    num_freeze = 10 # backbone
    
    model = trainer.model
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")


if __name__ == '__main__':
    
    with open(CFG_PATH / 'train.yaml', 'r') as f:
        cfgs = yaml.safe_load(f)
    
    data_cfg:Path = CFG_PATH / cfgs['data']
    cfgs['data'] = str(data_cfg.resolve())
    
    trainer = MyDetectionTrainer(overrides=cfgs)
    # trainer.add_callback("on_train_start", freeze_layer)
    trainer.train()