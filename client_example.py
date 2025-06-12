import requests
import base64
import os

screenshot_img_path = os.getcwd() + "/ScreenshotTest.png"

with open(screenshot_img_path, "rb") as img_file:
    base64_string = base64.b64encode(img_file.read()).decode("utf-8")

response = requests.post("http://localhost:5100/predict", json={"image": base64_string})
print(response)
print(response.json())