import numpy as np
import torch, os, json, io, cv2, time
from ultralytics import YOLO

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    env = os.environ
    model = YOLO("/opt/ml/model/" + env['YOLOV8_MODEL'])
    print(f"Loaded model - {env['YOLOV8_MODEL']}")
    return model

def input_fn(request_body, request_content_type):
    print("Executing input_fn from inference.py ...")
    
    if request_content_type.startswith("image/"):  # 'image/png', 'image/jpeg', 'image/jpg', etc.
        img = cv2.imdecode(np.frombuffer(request_body, dtype=np.uint8), flags=-1)
    elif request_content_type == "application/x-npy":
        # If you forget to explicitly set your PyTorchPredictor's `serializer` in SageMaker Python
        # SDK (invoking the endpoint from the notebook), your input (JPEG/PNG) image bytes will be
        # wrapped in a NumPy array. It's possible to read those requests like this:
        jpg_original = np.load(io.BytesIO(request_body), allow_pickle=True)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=-1)
    else:
        raise Exception("Unsupported content type: " + request_content_type)
    return img
    
def predict_fn(input_data, model):
    print("Executing predict_fn from inference.py ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        result = model(input_data)
    return result
        
# def output_fn(prediction_output, content_type):
#     print("Executing output_fn from inference.py ...")
#     infer = {}
#     for result in prediction_output:
#         if 'boxes' in result.keys:
#             infer['boxes'] = result.boxes.numpy().data.tolist()
#         if 'masks' in result.keys:
#             infer['masks'] = result.masks.numpy().data.tolist()
#         if 'keypoints' in result.keys:
#             infer['keypoints'] = result.keypoints.numpy().data.tolist()
#         if 'probs' in result.keys:
#             infer['probs'] = result.probs.numpy().data.tolist()
#     return json.dumps(infer)

def output_fn(prediction, content_type):
    print("Executing output_fn from inference.py ...")
    return prediction[0].tojson()