# sony-image-classifier
Classifies real life images from screenshots

## Running the service
1. Run service by calling \n
`python3 image-classifier-service.py [port number]`
2. Image input can be passed in by calling \n
`response = requests.post("http://localhost:[port number]/predict", json={"image": [base64_string]})`
3. Output format
`response.json()`

{
   "predicted_class":String,
   "real_probability":String,
   "screenshot_probability":String,
   "sexual":{
      "drawings":String,
      "hentai":String,
      "neutral":String,
      "porn":String,
      "sexy":String
   }
}
The predicted class of the input image will be returned as a String ('screenshot' or 'real'). The probability of a real image and screenshot are also given as floats casted as Strings. In the case of a screenshot classification, the sexual classification will also be given with scores in the 5 categories of 'drawings', 'hentai', 'neutral', 'porn', and 'sexy' (also floats casted as Strings). In the case of a real life classification, the sexual classification will not be given (Null).

## Example
1. Run service \n
`python3 image-classifier-service.py 5100`

2. Run client
`python3 client_example.py`

3. Expected output
{
   "predicted_class":"screenshot",
   "real_probability":"0.0031911730766296387",
   "screenshot_probability":"0.9968088269233704",
   "sexual":{
      "drawings":"0.09492923",
      "hentai":"0.04795105",
      "neutral":"0.787289",
      "porn":"0.025196178",
      "sexy":"0.044634487"
   }
}

## Training
Train folder contains training code that produced [text](mobile-net-v3-GAMENET700K-epochs-3-BS-512-LR-1e-3.pt)