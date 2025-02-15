from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth
import torch
import scipy.io as sio

DEVICE = 'cpu'
image_path = "assets/example_images/image.jpg"
prompt_depth_path = "assets/example_images/arkit_depth.png"
image = load_image(image_path).to(DEVICE)
prompt_depth = load_depth(prompt_depth_path).to(DEVICE) # 192x256, ARKit LiDAR depth in meters

model_path = "prompt_depth_anything_vits.onnx"

model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vits",model_kwargs = {"encoder":"vits"}).to(DEVICE).eval()
depth = model.predict(image, prompt_depth) # HxW, depth in meters
sio.savemat("prompt_depth_anything_vits_gt.mat", 
    {"image":image,
    "prompt_depth":prompt_depth,
    "depth":depth})
    
torch.onnx.export(
    model,
    (image, prompt_depth),
    model_path,
    input_names=["image","prompt_depth"],
    output_names=["depth"],
    opset_version=17,
    dynamic_axes={'image':{0: 'batch_size', 2: 'height', 3: 'width'},
                'prompt_depth':{0: 'batch_size', 2: 'height', 3: 'width'}}
)

import onnxruntime as ort
session = ort.InferenceSession(model_path)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Run inference
depth_onnx = session.run(None, {"image": to_numpy(image),"prompt_depth": to_numpy(prompt_depth)})

save_depth(depth, prompt_depth=prompt_depth, image=image)
