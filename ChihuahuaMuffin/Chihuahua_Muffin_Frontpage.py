import streamlit as st

st.set_page_config(
    page_title="ChihuahuaMuffin",
)

st.title('Chihuahua vs Blueberry Muffin Classification')
st.write("by Eric Ching")
st.write("# Welcome to my neural network project and demo!")
st.write("This was a pet project I completed over Fall/Winter 2023. It is a simple binary image classifier written in PyTorch with a Streamlit web interface.")
st.write("This was trained on a Kaggle dataset that can be found here: https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification/code.")
st.write("Thanks to Samuel Cortinhas for the dataset, truly the hardest part of data science.")
st.write("All code can be found at https://github.com/EricEChing/ChihuahuaMuffin.")

st.sidebar.success("Select an option above.")


st.image("/mount/src/chihuahuamuffin/ChihuahuaMuffin/archive(1)/chimufgrid.jpg")
st.image("archive(1)/DaThinka.jpg")


# command to run in browser
# streamlit run /Users/ericching/Documents/GitHub/ChihuahuaMuffin/ChihuahuaMuffin/Chihuahua_Muffin_Frontpage.py
