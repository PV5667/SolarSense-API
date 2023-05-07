import streamlit as st
import torch
import torchvision
import torch.nn as nn
import warnings
import geemap.foliumap as geemap
import geojson
import requests
from PIL import Image
import numpy as np
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

token_name = "GEE_TOKEN"
@st.cache(persist=True)
def ee_authenticate(token_name="GEE_TOKEN"):
    geemap.ee_initialize(token_name=token_name)

ee_authenticate(token_name=token_name)

warnings.filterwarnings("ignore")

st.title("Solar Panel Detection")
st.write("This is a demo of the solar panel detection application.")

def load_classification_model(model_path):
    #model = torchvision.models.resnet50()
    #model.fc = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))
    model = torchvision.models.efficientnet_b3(weights="EfficientNet_B3_Weights.IMAGENET1K_V1")
    model.classifier = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_segmentation_model(model_path):
    num_classes = 1
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier = DeepLabHead(2048, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

run_model = False

def find_bounds(coordinates):
    minx = coordinates[0][0]
    miny = coordinates[0][1]
    maxx = coordinates[2][0]
    maxy = coordinates[2][1]
    return minx, miny, maxx, maxy

upload_choice = st.selectbox("Upload a GeoJSON or manually select an area", ("Upload", "Select"))

if upload_choice == "Upload":
    geoj = st.file_uploader("Upload a GeoJSON", type="geojson")

    if geoj is not None:
        gj = geojson.load(geoj)
        coordinates = gj["features"][0]["geometry"]["coordinates"][0]
        m = geemap.Map(
            basemap="HYBRID",
            plugin_Draw=True,
            Draw_export=True,
            locate_control=True,
            plugin_LatLngPopup=True,
        )
        m.zoom_to_bounds(find_bounds(coordinates))
        m.to_streamlit(height=500)

        run_model = st.button("Run Model")

elif upload_choice == "Select":
    st.markdown("1. Select an area on the map through the sidebar features and click export to get a geojson file.")
    st.markdown("2. Next, upload the geojson file to the app.")

    m = geemap.Map(
        basemap="HYBRID",
        plugin_Draw=True,
        Draw_export=True,
        locate_control=True,
        plugin_LatLngPopup=True,
    )
    m.to_streamlit(height=500)

def split_into_patches(image):
    """
    Image is a numpy array.
    """
    images = []
    for r in range(0,image.shape[0], 400):
        for c in range(0,image.shape[1], 400):
            images.append(image[r:r+400, c:c+400,:])
    return images

def preprocess_classification(image):
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]
    transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=pretrained_means,
                                    std=pretrained_stds)
                       ])
    image = transforms(image)
    return image

def preprocess_segmentation(image):
    transforms = A.Compose([
                A.Normalize(
                  mean=[0, 0, 0],
                  std=[1.0, 1.0, 1.0],
                ),
                ToTensorV2(),
            ])
    transformed = transforms(image=image)
    image_out = transformed["image"] / 255.0
    return image_out

if run_model:
    coordinates = find_bounds(coordinates)
    centerlat = (coordinates[1] + coordinates[3])/2
    centerlon = (coordinates[0] + coordinates[2])/2
    zoom=18
    #URL = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{list(find_bounds(coordinates))}/1200x1200?access_token="
    access_token = st.secrets["MAPBOX_TOKEN"]
    URL = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{centerlon},{centerlat},{zoom},0/1200x1200?access_token={access_token}"
    r = requests.get(url = URL)
    file = open("satellite_img.png", "wb")
    file.write(r.content)
    file.close()
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        st.image("satellite_img.png", use_column_width=True, caption="Input Image into Model")
    image = np.array(Image.open("satellite_img.png").convert("RGB"))
    print(image.shape)
    images = split_into_patches(image)
    classification_model = load_classification_model("solar_classification 3.pt")
    segmentation_model = load_segmentation_model("solar_segmentation.pt")
    final_img = np.array(Image.open("satellite_img.png").convert("RGB"))
    final_patches = []
    for i in range(len(images)):
        patch = images[i]
        patch_classify = preprocess_classification(patch)
        patch_classify = patch_classify.unsqueeze(0)
        outputs = classification_model(patch_classify)
        _, predicted = torch.max(outputs, 1)
        if predicted.item() == 1:
            with col2:
                st.write("Solar Panel Detected")
                st.image(patch, use_column_width=True)
            patch_segmentation = preprocess_segmentation(patch)
            patch_segmentation = patch_segmentation.unsqueeze(0)
            pred = segmentation_model(patch_segmentation)
            mask = torch.sigmoid(pred["out"])[0].detach().numpy()[0]
            thresh = 0.05
            mask[mask < thresh] = 0.0
            mask[mask > thresh] = 1.0
            indices = np.where(mask == 1.0)
            patch[indices] = (100, 255, 100)
            with col2:
                st.image(patch, use_column_width=True, caption="Segmentation Mask")
            final_patches.append(patch)
        else: 
            final_patches.append(patch)
    for r in range(0,image.shape[0], 400):
        for c in range(0,image.shape[1], 400):
            final_img[r:r+400, c:c+400, :] = final_patches.pop(0)
    with col2:
        st.image(final_img, use_column_width=True, caption="Final Image")





    
    



 



