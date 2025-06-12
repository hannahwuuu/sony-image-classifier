# sony-image-classifier
Classifies real life images from screenshots

## Installation
1. Clone the repo  
`git clone git@github.com:hannahwuuu/sony-image-classifier.git`
2. Install necessary libraries  
`pip install requirements.txt`

## Running the service
1. Run service by calling  
`python3 image-classifier-service.py [port number]`
2. Image input can be passed in by calling  
`response = requests.post("http://localhost:[port number]/predict", json={"image": [base64_string]})`
3. Output format  
`response.json()`
<pre> {
   "predicted_class":String, # 'screenshot' or 'real'
   "real_probability":String,
   "screenshot_probability":String,
   "sexual":{ 
      "drawings":String,
      "hentai":String,
      "neutral":String,
      "porn":String,
      "sexy":String
   } # will be Null if predicted class is screenshot
}</pre>

## Example
1. Run service  
`python3 image-classifier-service.py 5100`

2. Run client  
`python3 client_example.py`

3. Expected output  
<pre>{
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
}</pre>

## Training
Train folder contains training code that produced [weights](mobile-net-v3-GAMENET700K-epochs-3-BS-512-LR-1e-3.pt) used in the service.