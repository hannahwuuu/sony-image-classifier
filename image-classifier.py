import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from keras.layers import TFSMLayer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import base64
import io
from game_net_v1 import GameNetV1

def transform_image(img_bytes):

  IMG_SIZE = (224, 224)

  base_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
  )

  try:
      img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
      img = transforms.Resize(IMG_SIZE)(img)
      img = base_transform(img)
      processed_img = img.numpy()
      processed_img = torch.tensor(processed_img, dtype=torch.float32)
      return processed_img
  except:
    print(f'Failed to open {img_bytes}. Corrupted file.')

# Create a function to check if an image is blank using Flood Fill
def is_blank_image(img_bytes):
  # Load image
  np_arr = np.frombuffer(img_bytes, np.uint8)
  img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

  # Make sure the image is loaded
  if img is None:
    print(f"Image unable to load for {img_bytes}")
    return False

  # Get the dimensions of the image
  height, width = img.shape

  # Ensure that there is padding for Flood Fill
  padding = np.zeros((height + 2, width + 2), np.uint8)

  # Flood Fill from (0,0)
  _, img_filled, _, _ = cv2.floodFill(img.copy(), padding, (0, 0), 255)

  # Check if the image is completely filled i.e. it's blank
  if np.all(img_filled == 255) or np.all(img_filled == 0):
    return True
  else:
    return False

def hsv_classify_image(image_bytes, vg_threshold=0.95, real_threshold=0.07):
    """
    Classifies a single image as video game (1) or real-life (0) based on HSV distribution.

    Args:
        image_path (str): Path to the image file
        vg_threshold (float): Confidence threshold for video game classification (default: 0.95)
        real_threshold (float): Confidence threshold for real-life classification (default: 0.07)

    Returns:
        int: 1 if confidently video game, 0 if confidently real-life, None if uncertain
    """

    def extract_hsv_features(img_bytes):
        """Extract HSV statistical features from an image."""
        try:
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot load image")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            return {
                "mean_h": np.mean(h), "mean_s": np.mean(s), "mean_v": np.mean(v),
                "std_h": np.std(h), "std_s": np.std(s), "std_v": np.std(v)
            }
        except Exception as e:
            print(f"Failed to process {img_bytes}: {e}")
            return None

    def predict_confidence(features):
        """Calculate confidence score based on HSV features."""
        score = 0

        # Value (brightness) rules
        if features["mean_v"] > 130:
            score += 0.5
        elif features["mean_v"] < 90:
            score -= 0.3

        # Value standard deviation rules
        if features["std_v"] > 45:
            score += 0.3
        elif features["std_v"] < 30:
            score -= 0.3

        # Hue rules
        if features["mean_h"] < 60:
            score += 0.2
        elif features["mean_h"] > 90:
            score -= 0.2

        return round(np.clip((score + 1) / 2, 0, 1), 3)

    # Extract features from the image
    features = extract_hsv_features(image_bytes)
    if features is None:
        return None

    # Calculate confidence score
    confidence = predict_confidence(features)

    # Make classification based on thresholds
    if confidence >= vg_threshold:
        return 1  # Video game
    elif confidence <= real_threshold:
        return 0  # Real-life
    else:
        return None  # Uncertain - let other methods decide

def sexual_prediction(image_path):
    sexual_model = TFSMLayer('/content/gdrive/MyDrive/mobilenet_v2_140_224', call_endpoint='serving_default')

    labels = ["drawings", "hentai", "neutral", "porn", "sexy"]

    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    image = preprocess_image(image_path)

    output = sexual_model(image)

    probabilities = output['prediction'].numpy()[0]

    results = dict(zip(labels, probabilities))

    print("Class probability distribution:")
    for label, probability in results.items():

        print(f"{label}: {probability:.4f}")

    predicted_class = labels[np.argmax(probabilities)]
    print(f"\nPredicted class: {predicted_class}")

def predict(model, image, return_proba):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    with torch.no_grad():
        if torch.cuda.is_available():
            image = image.cuda()
        pred = model(image.unsqueeze(0)).squeeze()
        prob = torch.sigmoid(pred)
        if return_proba:
          return prob
        final_prediction = (prob >= 0.55).float()

    return final_prediction
   
def final_prediction(image_bytes, return_proba=False):
  
  # transform image
  image_bytes = base64.b64decode(image_bytes)
  image = transform_image(image_bytes)

  blank_img = is_blank_image(image_bytes)

  # check if it is a blank
  if blank_img:
    return 1

  hsv_result = hsv_classify_image(image_bytes)
  if hsv_result is not None:
      return hsv_result

  final_prediction = predict(model, image, return_proba)
  ## TODO check whether we want sexual prediction here
  return final_prediction.item()

# model, train_losses, avg_train_losses, test_losses, avg_test_losses = run_training(hp_base, VGG, "model_DenseNet161_1_NM")
model = GameNetV1()
if torch.cuda.is_available():
    model.load_state_dict(torch.load(os.getcwd() + "/mobile-net-v3-GAMENET700K-epochs-3-BS-512-LR-1e-3.pt", map_location=torch.device('cuda:0')))
else:
   model.load_state_dict(torch.load(os.getcwd() + "/mobile-net-v3-GAMENET700K-epochs-3-BS-512-LR-1e-3.pt", map_location=torch.device('cpu')))

screenshot_img_path = os.getcwd() + "/ScreenshotTest.png"

with open(screenshot_img_path, "rb") as img_file:
    base64_string = base64.b64encode(img_file.read())

print(final_prediction(base64_string))