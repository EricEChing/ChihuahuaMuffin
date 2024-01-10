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

if st.button("start!"):
    game_end = False
    human_count = 0
    robot_count = 0
    while game_end == False:
        next_image, label = next(iter(example_loader))
        st.image(transforms.ToPILImage()(next_image.squeeze()))
        output = model(next_image)
        output = round(float(output))

        button_end = False

        if output == label:
            robot_count += 1

        while button_end == False:
            if st.button("Chihuahua") and label == 0:
                human_count += 1
                button_end = True
            elif st.button("Muffin") and label == 1:
                human_count += 1
                button_end = True
            time.sleep(60)

        if human_count == 10 and robot_count == 10:
            st.write("Tie!")
            game_end = True
        elif robot_count == 10:
            st.write("You Lose!")
            game_end = True
        elif human_count == 10:
            st.write("You win!")
            game_end = True
