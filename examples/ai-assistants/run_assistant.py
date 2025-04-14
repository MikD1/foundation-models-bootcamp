import os

from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()
sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    auth=os.getenv("YANDEX_API_KEY"),
)
assistant = sdk.assistants.get(open('assistant_id.txt').read())

query = 'Что такое гид цу'

text_index_thread = sdk.threads.create()
text_index_thread.write(query)
run = assistant.run(text_index_thread)

result = run.wait().message
for part in result.parts:
    print(part)
