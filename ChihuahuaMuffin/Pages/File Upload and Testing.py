import streamlit as st
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from Non_Interface_Internals.ChihuahuaMuffin import Classifier
from PIL import Image

transform = transforms.Compose([


    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

exampleImagePath = 'archive(1)/train'
example_set = ImageFolder(root=exampleImagePath,transform=transform)

example_loader = DataLoader(example_set,batch_size=1,shuffle=True)

def writeResult(inputTensor, output):
    result = float(output.view(-1))
    if round(result)==1:
        percentage = float(output.view(-1))
        st.write("I'm " + str(round(percentage*100, 2)) + '%' + " sure its a blueberry muffin!")
    else:
        percentage = 1 - float(output.view(-1))
        st.write("I'm " + str(round(percentage*100, 2)) + '%' + " sure its a Chihuahua!")

    st.image(transforms.ToPILImage()(inputTensor.squeeze()))


os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

model = Classifier()
model.load_state_dict(torch.load('Non_Interface_Internals/ChihuahuaMuffin.pth'))
model.eval()  # Set the model to evaluation mode

st.title("Dedicated File Upload and Testing")

uploaded_file = st.file_uploader("Choose a blueberry muffin or Chihuahua JPEG or PNG file")

if uploaded_file is not None:
    inputTensor = transform(Image.open(uploaded_file))

    with torch.inference_mode():
        output = model(inputTensor)
    
    writeResult(inputTensor.view(1, 3, 256, 256),output)

    
if st.button("Choose a random Chihuahua:"):
    inputTensor = next(iter(example_loader))
    while round(int(inputTensor[1])) != 0:
        inputTensor = next(iter(example_loader))
    inputTensor = inputTensor[0]

    with torch.inference_mode():
        output = model(inputTensor)

    writeResult(inputTensor, output)

if st.button("Choose a random blueberry muffin:"):
    inputTensor = next(iter(example_loader))
    while round(int(inputTensor[1])) != 1:
        inputTensor = next(iter(example_loader))
    inputTensor = inputTensor[0]

    with torch.inference_mode():
        output = model(inputTensor)

    writeResult(inputTensor, output)



    