import streamlit as st
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from ChihuahuaMuffin import Classifier
import os
from PIL import Image

transform = transforms.Compose([


    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

# command to run in browser
# streamlit run /Users/ericching/Documents/ChihuahuaMuffin/chiMufInterface.py 

os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

model = Classifier()
model.load_state_dict(torch.load('ChihuahuaMuffin.pth'))
model.eval()  # Set the model to evaluation mode

st.title('Chihuahua vs Blueberry Muffin Classification')
st.write("by Eric Ching")

st.write("Dedicated File Upload and Testing")

uploaded_file = st.file_uploader("Choose a blueberry muffin or Chihuahua JPEG or PNG file")

if uploaded_file is not None:
    inputTensor = transform(Image.open(uploaded_file))

    with torch.inference_mode():
        output = model(inputTensor)

    result = float(output.view(-1))
    if round(result)==1:
        percentage = float(output.view(-1))
        st.write("I'm " + str(round(percentage*100, 2)) + '%' + " sure its a blueberry muffin!")
    else:
        percentage = 1 - float(output.view(-1))
        st.write("I'm " + str(round(percentage*100, 2)) + '%' + " sure its a Chihuahua!")

st.write("Human vs. Machine")

exampleImagePath = 'archive(1)/train'
example_set = ImageFolder(root=exampleImagePath,transform=transform)

example_loader = DataLoader(example_set,batch_size=1,shuffle=True)
        
global robot_count
global human_count 
global label

robot_count = 0
human_count = 0

game_image = st.empty()

example_image, label = next(iter(example_loader))

with game_image.container():
    st.image(transforms.ToPILImage()(example_image.view(3,256,256)))

label = int(label.view(-1))

output = model(example_image)

if round(float(output.view(-1)))==label:
    robot_count += 1

userPoints = st.empty()
robotPoints = st.empty()
with userPoints.container():
    st.write("Your Points: " + str(human_count))
with robotPoints.container():
    st.write("Robot Points: " + str(robot_count)) 

if st.button("Chihuahua"):
    if label==0:
        human_count +=1
    game_image.empty()
    userPoints.empty()
    robotPoints.empty()


    example_image, label = next(iter(example_loader))

    with game_image.container():
        st.image(transforms.ToPILImage()(example_image.view(3,256,256)))

    label = int(label.view(-1))

    output = model(example_image)

    if round(float(output.view(-1)))==label:
        robot_count += 1

    with userPoints.container():
        st.write("Your Points: " + str(human_count))
    with robotPoints.container():
        st.write("Robot Points: " + str(robot_count)) 



if st.button("Blueberry Muffin"):
    if label==1:
        human_count +=1
    game_image.empty()
    userPoints.empty()
    robotPoints.empty()

    example_image, label = next(iter(example_loader))

    with game_image.container():
        st.image(transforms.ToPILImage()(example_image.view(3,256,256)))

    label = int(label.view(-1))

    output = model(example_image)

    if round(float(output.view(-1)))==label:
        robot_count += 1

    with userPoints.container():
        st.write("Your Points: " + str(human_count))
    with robotPoints.container():
        st.write("Robot Points: " + str(robot_count)) 


if robot_count == 10:
    st.write("You Lose! You are dumber than 26 lines of code!")

if human_count == 10:
    st.write("You Win! You are smarter than 26 lines of code!")

