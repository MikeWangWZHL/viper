import os
from omegaconf import OmegaConf

# The default
config_names = os.getenv('CONFIG_NAMES', None)
config_dir = os.getenv('CONFIG_DIR', 'configs')
print(f'loading config from config_dir: {config_dir}')
if config_names is None:
    config_names = 'my_config'  # Modify this if you want to use another default config

configs = [OmegaConf.load(os.path.join(config_dir, 'base_config.yaml'))]

if config_names is not None:
    for config_name in config_names.split(','):
        configs.append(OmegaConf.load(os.path.join(config_dir, f'{config_name.strip()}.yaml')))

# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.unsafe_merge(*configs)

