# inference-repo/scripts/update_model_version.py
import yaml
import argparse


def update_model_version(config_file, model_version):
    """Update model version in configuration file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update model version in config
    if 'model' not in config:
        config['model'] = {}

    config['model']['version'] = model_version
    config['model']['path'] = f'models/{model_version}/model.pth'

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Updated {config_file} to model version: {model_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--config-file', required=True)
    args = parser.parse_args()

    update_model_version(args.config_file, args.model_version)