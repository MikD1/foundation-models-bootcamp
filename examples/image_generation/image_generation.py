import os
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()
sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    auth=os.getenv("YANDEX_API_KEY"),
)
model = sdk.models.image_generation('yandex-art')
model.configure(
    seed=int(round(datetime.now().timestamp())),
)

prompt = "Милый пушистый котенок спит на спине. Octane render,f/2.8, ISO 200"
messages = [
    {"weight": 1, "text": prompt},
]

operation = model.run_deferred(messages)
result = operation.wait()
image = Image.open(BytesIO(result.image_bytes))
image.show()
