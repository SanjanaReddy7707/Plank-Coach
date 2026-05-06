import urllib.request, os
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = "/app/pose_landmarker_lite.task"
print("Downloading pose model...")
urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
print(f"Done. {os.path.getsize(MODEL_PATH)} bytes")