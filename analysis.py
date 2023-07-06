#%%
import requests
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import numpy as np
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import rasterio.features
import shapely.wkt
import geopandas as gpd
from shapely.geometry import Polygon
import geojson
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoicHY1NjY3IiwiYSI6ImNsZGFtOHVoejBiZ2Mzb3A2djgyaDl1OGEifQ.FSssERk7wLiG1fDpen0iXA'

IMG_HEIGHT = 400
IMG_WIDTH =  400
mps_device = "cpu" #torch.device("mps")

PATCH_SIZE = 400

def bbox_from_coords(coords):
    bbox = [coords[0][0], coords[0][1], coords[2][0],coords[2][1]]
    return bbox

def call_api(bbox):
    api_string = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]/{IMG_HEIGHT}x{IMG_WIDTH}?access_token={MAPBOX_ACCESS_TOKEN}"
    #print(api_string)
    response = requests.get(api_string)
    if response.status_code == 200:
        path = "analysis.png"
        with open(path, 'wb') as f:
            f.write(response.content)
            print("Mapbox API called")
        return True
    else:
        print(f"Error occured code {response.status_code}")
        return False


# %%
"""
def load_classification_model(model_path):
    model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
"""

def load_classification_model(model_path):
    #model = torchvision.models.efficientnet_b3(weights="EfficientNet_B3_Weights.IMAGENET1K_V1")
    #model.classifier = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    print("classification model loading")
    model = torchvision.models.inception_v3(weights=None)
    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(mps_device)
    return model

def load_segmentation_model(model_path):
    print("segmentation model loading")
    num_classes = 1
    model = deeplabv3_resnet50(weights="DeepLabV3_ResNet50_Weights.DEFAULT")
    model.classifier = DeepLabHead(2048, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(mps_device)
    return model

classification_model = load_classification_model("gsolar_classification_affine5.pt")

segmentation_model = load_segmentation_model("solar_segmentation.pt")

pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

#pretrained_means = [0.0, 0.0, 0.0]
#pretrained_stds = [1.0, 1.0, 1.0]

classification_transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=pretrained_means,
                                    std=pretrained_stds)
                       ])
segmentation_transforms = A.Compose([
                A.Normalize(
                  mean=[0, 0, 0],
                  std=[1.0, 1.0, 1.0],
                ),
                ToTensorV2(),
            ])

def preprocess_classification(image):
    image = classification_transforms(image)
    return image

def preprocess_segmentation(image):
    transformed = segmentation_transforms(image=image)
    image_out = transformed["image"] / 255.0
    return image_out

def split_into_patches(image):
    """
    Image is a numpy array.
    """
    images = []
    for r in range(0,image.shape[0], PATCH_SIZE):
        for c in range(0,image.shape[1], PATCH_SIZE):
            #print(r, c)
            images.append(image[r:r+PATCH_SIZE, c:c+PATCH_SIZE,:])
    return images

def run_analysis_with_patches():
    image = np.array(Image.open("analysis.png").convert("RGB"))
    images = split_into_patches(image)
    #print(images)
    det_polys = []
    for patch in images:
        image_processed = preprocess_classification(patch).unsqueeze(0).to(mps_device)
        outputs = classification_model(image_processed)
        _, predicted = torch.max(outputs, 1)
        del image_processed
        #plt.imshow(patch)
        #print(predicted.item())
        if predicted.item() == 1:
            print("Solar panel detected")
            #plt.imshow(patch)
            image_processed = preprocess_segmentation(patch).unsqueeze(0).to(mps_device)
            mask, polygons = run_segmentation(image_processed, segmentation_model)
            del image_processed
            det_polys.append(polygons)
    out = [poly for polylist in det_polys for poly in polylist]
    del det_polys
    return out if len(out) > 0 else None

def run_analysis():
    image = np.array(Image.open("analysis.png").convert("RGB"))
    #print(images)
    image_processed = preprocess_classification(image).unsqueeze(0).to(mps_device)
    outputs = classification_model(image_processed)
    _, predicted = torch.max(outputs, 1)
    del image_processed
        #plt.imshow(patch)
        #print(predicted.item())
    if predicted.item() == 1:
        print("Solar panel detected")
        #plt.imshow(patch)
        image_processed = preprocess_segmentation(image).unsqueeze(0).to(mps_device)
        del image
        mask, polygons = run_segmentation(image_processed, segmentation_model)
        del image_processed
        out = [poly for poly in polygons]
        return out if len(out) > 0 else None
    return None

def run_segmentation(processed_image, model):
    outputs = model(processed_image)['out']
    outputs = torch.sigmoid(outputs)
    mask = outputs[0].cpu().detach().numpy()[0]
    thresh = 0.05
    mask[mask < thresh] = 0.0
    mask[mask > thresh] = 1.0
    #indices = np.where(mask == 1.0)
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    polygons = []
    for shape, val in shapes:
        s = shapely.geometry.shape(shape).exterior
        polygons.append(s)
    return mask, polygons

def convert_to_zoom(bbox):
    centerlat = round((bbox[1] + bbox[3])/2, 4)
    centerlon = round((bbox[0] + bbox[2])/2, 4)
    zoom=18.1
    return centerlat, centerlon, zoom

def convert_to_lat_lon(top_left, bottom_right, x, y, coords):
    delta = np.array(bottom_right - top_left)
    shape = np.array((x, y))
    pixel_sizes = delta / shape
    return coords * pixel_sizes + top_left

def extract_tl_br(bbox):
    top_left = np.array([bbox[0], bbox[1]])
    bottom_right = np.array([bbox[2], bbox[3]])
    return top_left, bottom_right

def main(selection):
    polygon_geoms = []
    for i in tqdm(range(len(selection["features"]))):
        coords = selection["features"][i]["geometry"]["coordinates"][0]
        bbox = bbox_from_coords(coords)
        call_api(bbox)
        print("Running Analysis Now...")
        polygons = run_analysis()
        if polygons is not None:
            top_left, bottom_right = extract_tl_br(bbox)
            for polygon in polygons:
                #coords = polygon.coords.xy
                coords = np.dstack((polygon.coords.xy[0], IMG_HEIGHT - np.array(polygon.coords.xy[1]))).squeeze()
                #print(coords)
                lat_lon_coords = convert_to_lat_lon(top_left, bottom_right, IMG_HEIGHT, IMG_WIDTH, coords)
                polygon_geoms.append(Polygon(lat_lon_coords))
        del polygons

    if polygon_geoms is not None:
        out_gdf = gpd.GeoDataFrame(geometry=polygon_geoms, crs="EPSG:4326")
    del polygon_geoms
    #with open('panels.geojson' , 'w') as file:
    #    file.write(out_gdf.to_json())

    return out_gdf.to_json()

#selection = geojson.load(open('data.json'))
#main(selection)
# %%
