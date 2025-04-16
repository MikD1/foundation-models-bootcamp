import os

from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

message = "Нейросети для творчества и заработка"

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

best_prediction = result.predictions[0]
for prediction in result.predictions:
    if prediction.confidence > best_prediction.confidence:
        best_prediction = prediction

print(best_prediction.label)