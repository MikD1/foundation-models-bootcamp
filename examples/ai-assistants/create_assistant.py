import os

from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    TextSearchIndexType
)

load_dotenv()
sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    auth=os.getenv("YANDEX_API_KEY"),
)

file = sdk.files.upload("./data/grant.md")
operation = sdk.search_indexes.create_deferred([file], index_type=TextSearchIndexType())

text_index = operation.wait()
text_tool = sdk.tools.search_index(text_index)

model = sdk.models.completions("yandexgpt", model_version="rc")
assistant = sdk.assistants.create(model, tools=[text_tool])

print(assistant.id)
with open('assistant_id.txt', 'w', encoding='utf-8') as f:
    f.write(assistant.id)
