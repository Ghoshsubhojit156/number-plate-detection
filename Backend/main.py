# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from ultralytics import YOLO
# import easyocr
# import numpy as np
# import cv2
# import uvicorn
# import base64

# app = FastAPI()

# # Load Model
# model = YOLO("C:/Users/subho/OneDrive - uem.edu.in/Desktop/NUmber_Plate_Detection_New/Trained_Model/Number_plate.pt")

# # OCR Reader
# reader = easyocr.Reader(['en'])

# @app.get("/")
# def home():
#     return {"message": "Number Plate Detection API Running"}

# @app.post("/detect")
# async def detect(file: UploadFile = File(...)):

#     contents = await file.read()

#     nparr = np.frombuffer(contents, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     results = model(image)

#     detections = []

#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()

#         for box in boxes:

#             x1, y1, x2, y2 = map(int, box)

#             cropped_plate = image[y1:y2, x1:x2]

#             ocr_result = reader.readtext(cropped_plate)

#             plate_text = ""

#             if len(ocr_result) > 0:
#                 plate_text = ocr_result[0][1]

#             # Draw Bounding Box
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

#             cv2.putText(
#                 image,
#                 plate_text,
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0,255,0),
#                 2
#             )

#             detections.append({
#                 "bbox": [x1, y1, x2, y2],
#                 "text": plate_text
#             })

#     # Encode Image
#     _, buffer = cv2.imencode(".jpg", image)
#     encoded_image = base64.b64encode(buffer).decode("utf-8")

#     return JSONResponse({
#         "detections": detections,
#         "image": encoded_image
#     })

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from ultralytics import YOLO

from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel

from PIL import Image

import numpy as np
import cv2
import base64
import torch
import uvicorn
import os
Base_Dir=os.getcwd()
app = FastAPI()

print("Current Directory:", Base_Dir)
# =========================
# LOAD YOLO MODEL
# =========================
model = YOLO(os.path.join(Base_Dir, "Number_plate.pt"))

# =========================
# LOAD HUGGINGFACE OCR MODEL
# =========================
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-printed"
)

ocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

ocr_model.to(device)

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(image)

    detections = []

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            # Crop Plate
            cropped_plate = image[y1:y2, x1:x2]

            # Convert BGR → RGB
            cropped_plate_rgb = cv2.cvtColor(
                cropped_plate,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(cropped_plate_rgb)

            # OCR
            pixel_values = processor(
                images=pil_image,
                return_tensors="pt"
            ).pixel_values.to(device)

            generated_ids = ocr_model.generate(pixel_values)

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            # Draw Bounding Box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            cv2.putText(
                image,
                generated_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "text": generated_text
            })

    # Encode Output Image
    _, buffer = cv2.imencode(".jpg", image)

    encoded_image = base64.b64encode(
        buffer
    ).decode("utf-8")

    return JSONResponse({
        "detections": detections,
        "image": encoded_image
    })

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )