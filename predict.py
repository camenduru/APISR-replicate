import os, sys
from cog import BasePredictor, Input, Path
sys.path.append('/content/APISR-hf')
os.chdir('/content/APISR-hf')

import cv2, torch
from test_code.inference import super_resolve_img
from test_code.test_utils import load_grl, load_rrdb

def inference(img_path, model_name, output_path):
    try:
        weight_dtype = torch.float32
        if model_name == "4xGRL":
            weight_path = "pretrained/4x_APISR_GRL_GAN_generator.pth"
            generator = load_grl(weight_path, scale=4)
        elif model_name == "2xRRDB":
            weight_path = "pretrained/2x_APISR_RRDB_GAN_generator.pth"
            generator = load_rrdb(weight_path, scale=2)
        else:
            print("We don't support such Model")
        generator = generator.to(dtype=weight_dtype)
        super_resolved_img = super_resolve_img(generator, img_path, output_path=output_path, weight_dtype=weight_dtype, crop_for_4x=True)
        return super_resolved_img
    except Exception as error:
        print(f"global exception: {error}")

class Predictor(BasePredictor):
    def setup(self) -> None:
        print('setup')
    def predict(
        self,
        img_path: Path = Input(description="Image"),
        model_name: str = Input(choices=['4xGRL','2xRRDB'], default='4xGRL'),
    ) -> Path:
        output_image = inference(str(img_path), model_name, '/content/output_image.png')
        return Path('/content/output_image.png')