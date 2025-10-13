# inference-repo/scripts/download_model.py
import boto3
import os
import yaml
from pathlib import Path


def download_latest_model():
    session = boto3.Session(
        aws_access_key_id=os.getenv('YC_SA_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('YC_SA_SECRET_KEY')
    )

    s3 = session.client('s3', endpoint_url='https://storage.yandexcloud.net')

    # Получение информации о последней модели
    response = s3.list_objects_v2(Bucket='auto-transport-models', Prefix='models/')

    if 'Contents' not in response:
        print("No models found in bucket")
        return

    # Поиск последней модели
    latest_model = max(response['Contents'], key=lambda x: x['LastModified'])
    model_key = latest_model['Key']

    # Создание структуры папок
    model_version = Path(model_key).parent.name
    model_dir = Path(f"models/{model_version}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка модели
    s3.download_file('auto-transport-models', model_key, str(model_dir / 'model.pth'))
    print(f"Downloaded model: {model_key}")

    # Обновление симлинка на текущую модель
    current_link = Path("models/current")
    if current_link.exists():
        current_link.unlink()
    current_link.symlink_to(model_version)


if __name__ == "__main__":
    download_latest_model()