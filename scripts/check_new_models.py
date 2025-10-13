# inference-repo/scripts/check_new_models.py
import boto3
import os
import yaml
from datetime import datetime
import argparse
import sys


def get_current_version(version_file):
    """Get current deployed model version"""
    try:
        with open(version_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "v0.0.0"


def get_available_models():
    """Get list of available models from Object Storage"""
    session = boto3.Session(
        aws_access_key_id=os.getenv('YC_SA_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('YC_SA_SECRET_KEY')
    )

    s3 = session.client('s3', endpoint_url='https://storage.yandexcloud.net')

    try:
        response = s3.list_objects_v2(
            Bucket='auto-transport-models',
            Prefix='models/',
            Delimiter='/'
        )

        models = []
        for prefix in response.get('CommonPrefixes', []):
            model_path = prefix['Prefix'].rstrip('/')
            model_version = model_path.split('/')[-1]
            models.append(model_version)

        return sorted(models, reverse=True)
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def get_latest_model_version():
    """Get the latest model version"""
    models = get_available_models()
    return models[0] if models else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current-version-file', required=True)
    parser.add_argument('--output-new-version', default='latest')
    args = parser.parse_args()

    current_version = get_current_version(args.current_version_file)

    if args.output_new_version == 'latest':
        new_version = get_latest_model_version()
    else:
        new_version = args.output_new_version

    if not new_version:
        print("No models available")
        sys.exit(1)

    # Check if new version is available
    if new_version != current_version:
        print(f"New model available: {new_version} (current: {current_version})")
        print(f"::set-output name=new_model_available::true")
        print(f"::set-output name=model_version::{new_version}")
        print(f"::set-output name=model_path::models/{new_version}")
    else:
        print(f"Current model is up to date: {current_version}")
        print(f"::set-output name=new_model_available::false")


if __name__ == "__main__":
    main()