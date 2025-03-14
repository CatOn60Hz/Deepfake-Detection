
import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
import pandas as pd

warnings.filterwarnings("ignore")






DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,         #USE MTCNN FOR EVALUATION 
    device=DEVICE
).to(DEVICE).eval()




model = InceptionResnetV1(
    pretrained="vggface2",  #LOAD PRETAINED WEIGHTS 
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load(r"C:\Users\Arfan\Desktop\deepshit\trainedmodel.pth", map_location=torch.device('cpu')) #LOAD THE MODEL 
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()




def predict(input_image:Image.Image):                    

    """Predict the label of the input_image"""                                                                  
    face = mtcnn(input_image)                                                                 #MTCNN IS USED TO FIND THE FACE IN THE GIVEN IMAGE
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0)                                                                  
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)   #PICTURE IS RESIZED 
    

    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()                   
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0                                                                        #FACE IS NORMALIZED (A VALUE BETWEEN 0 AND 1 IS OBTAINED)
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()          

    target_layers=[model.block8.branch1[-1]]
    use_cuda = True if torch.cuda.is_available() else False
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)                                      #A Grad-CAM heatmap is generated to visualize important regions in the face image
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"                                                #The model outputs a confidence score between 0 and 1, which is interpreted as the likelihood of the face being "real" or "fake"
                                                                                                                                
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        
        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }
    return confidences, face_with_mask


gr.Markdown("# Permanent Title: My Gradio App")
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Image(label="Input Image", type="pil")
    ],
    outputs=[
        gr.outputs.Label(label="Class"),
        gr.outputs.Image(label="Face with Explainability", type="pil")
    ],
).launch(share=True)
