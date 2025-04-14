import os

from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

message = "Продам гараж, недорого, в лс"

load_dotenv()
sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    auth=os.getenv("YANDEX_API_KEY"),
)

model = sdk.models.text_classifiers("yandexgpt").configure(
    task_description="Определи тип сообщения",
    labels=["Спам", "Не спам"]
)

result = model.run(message)
prediction = max(result, key=lambda x: x.confidence)

print(prediction.label)
