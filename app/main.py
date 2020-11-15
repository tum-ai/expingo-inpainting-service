import numpy as np
import cv2

import uvicorn
import tensorflow as tf
import neuralgym as ng
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File
from fastapi import HTTPException
from inpaint.inpainting_model import InpaintCAModel


class PaintRequest(BaseModel):
    image: str
    mask: str


FLAGS = ng.Config('inpaint.yml')
MODEL_DIR = "../model_logs/places2"
MODEL = InpaintCAModel()
app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/inpaint/")
async def create_upload_file(request: PaintRequest):
    import base64
    import io
    from PIL import Image
    image = request.image
    mask = request.mask

    image = image.split(",", 1)[1]
    mask = mask.split(",", 1)[1]

    base64_decoded_image = base64.b64decode(image)

    image = Image.open(io.BytesIO(base64_decoded_image))
    image = np.array(image)

    base64_decoded_mask = base64.b64decode(mask)

    mask = Image.open(io.BytesIO(base64_decoded_mask))
    mask = np.array(mask)

    # mask is always PNG, image might have only 3 dimensions.
    mask = mask[:, :, :3]
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Catch weird error that image is turned if format is jpg and upright
    if image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
        image = np.flip(np.transpose(image, (1, 0, 2)), axis=1)
    if image.shape != mask.shape:
        raise HTTPException(
            status_code=400,
            detail=f"Image and Mask have unequal shape. {image.shape} vs {mask.shape}")

    # Image and Mask must be same dimension by now. Both have dimensions (x, y, 3)

    h, w, _ = image.shape
    grid = 8
    image = image[:h // grid * grid, :w // grid * grid, :]
    mask = mask[:h // grid * grid, :w // grid * grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    print(image.shape)
    print(mask.shape)
    input_image = np.concatenate([image, mask], axis=2)
    print(input_image.shape)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = MODEL.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(MODEL_DIR, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        cv2.imwrite("output.png", result[0])
    tf.reset_default_graph()
    #return FileResponse("output.png", media_type="image/png")
    with open("output.png", "rb") as image_file:
        image_string = "data:image/png;base64,{}".format(base64.b64encode(image_file.read()).decode())
    return {
        "image": image_string
    }


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
