# Model Evaluation (Spam Detection)

To evaluate your model, upload it to the evaluation portal running on the Playground VM. If you are not currently using the Playground VM, you can initialize it at the bottom of the page.

If you have the Playground VM running, you can use this Python script to upload your model from Jupyter directly. Once evaluated, if your model meets the required performance criteria, you will receive a flag value. This flag can be used to answer the question or verify the model's success.

```python
import requests
import json

# Define the URL of the API endpoint
url = "http://localhost:8000/api/upload"

# Path to the model file you want to upload
model_file_path = "spam_detection_model.joblib"

# Open the file in binary mode and send the POST request
with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

# Pretty print the response from the server
print(json.dumps(response.json(), indent=4))
```

If you are working from your own machine, ensure you have configured the HTB VPN to connect to the remote VM and spawned it. After connecting, access the model upload portal by navigating to http://<VM-IP>:8000/ in your browser and then uploading your model.

## VPN Servers

**Warning:** Each time you "Switch", your connection keys are regenerated and you must re-download your VPN connection file.

All VM instances associated with the old VPN Server will be terminated when switching to a new VPN server.
Existing PwnBox instances will automatically switch to the new VPN server.

## PROTOCOL

- UDP 1337
- TCP 443

## Connect to Pwnbox

Your own web-based Parrot Linux instance to play our labs.

**Pwnbox Location:** UK (87ms)

Terminate Pwnbox to switch location / 1 spawns left

Waiting to start...

Enable step-by-step solutions for all questions

## Questions

**Answer the question(s) below to complete this Section and earn cubes!**

**Target(s):** Click here to spawn the target system!

**+ 2** What is the flag you get from submitting a good model for evaluation?
