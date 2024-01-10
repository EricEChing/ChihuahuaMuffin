import pandas as pd
import numpy as np
import torch
import os
import streamlit as st
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from Non_Interface_Internals.ChihuahuaMuffin import Classifier
from PIL import Image
import time

transform = transforms.Compose([


    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

model = Classifier()
model.load_state_dict(torch.load('Non_Interface_Internals/ChihuahuaMuffin.pth'))
model.eval()  # Set the model to evaluation mode

st.title("Human vs. Machine")

exampleImagePath = 'archive(1)/train'
example_set = ImageFolder(root=exampleImagePath,transform=transform)

example_loader = DataLoader(example_set,batch_size=1,shuffle=True)

@st.cache(allow_output_mutation=True)
def get_human_count():
    return {"human_count": 0}

@st.cache(allow_output_mutation=True)
def get_robot_count():
    return {"robot_count": -1}

def generateNewImage():
    next_image, label = next(iter(example_loader))
    st.image(transforms.ToPILImage()(next_image.squeeze()))
    output = model(next_image)
    output = round(float(output))
    return output, label

robot_count = get_robot_count()
human_count = get_human_count()

output, label = generateNewImage()
if output == label:
    robot_count["robot_count"] += 1

if st.button("Chihuahua") and label == 0:
    human_count["human_count"] += 1

if st.button("Muffin") and label == 1:
    human_count["human_count"] += 1

if human_count["human_count"] == 10 and robot_count["robot_count"] == 10:
    st.header("Tie!")
elif robot_count["robot_count"] == 10:
    st.header("You Lose!")
elif human_count["human_count"] == 10:
    st.header("You win!")    

st.write("human: " + str(human_count["human_count"]))
st.write("AI: " + str(robot_count["robot_count"]))


