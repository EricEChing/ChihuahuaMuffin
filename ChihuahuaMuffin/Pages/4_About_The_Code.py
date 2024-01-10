import streamlit as st

model_toSTR = '''
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 256),  # Adjust based on your data dimensions
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x.view(1,-1))
        return x
'''

cycle_ToSTR = '''
for epoch in range(num_epochs):
    for index, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if index==10:
            break
        
    print('Epoch: {} Average Loss: {:.4f}'.format(epoch, loss / BATCH_SIZE))
'''

st.title("About Da Code:")
st.write("Can't imagine anyone cares, but I spent too much time formatting code for no one to see it.")
st.write("Here is the model code itself, with the only import dependency being torch.nn. [256x256x256] image tensor -> (0.0-1.0) confidence scalar")
st.code(body=model_toSTR)
st.write("Assuming the reader has a nonzero level of experience with PyTorch/Python ML, a relatively simple model is shown.")
st.write('''The only particulars of note are the Sigmoid function due to it being a binary classifier, 
the convolution layers and subsequent flattening for this image specific problem, and the use of leakyReLU. ''')
st.write("For those unfamiliar witbh leaky ReLU, it looks like this:")
st.image('archive(1)/leakyReLU.png')
st.write('where for x<0, y=ax with a typically being 0.01 as well as the default value for nn.leakyReLU.')
st.write("Anecdotally, I don't understand the math behind how it affects training/testing (reduces dead neurons/zero gradients), but I always end up using it over ReLU because the loss numbers are better.")

st.write("The testing/training cycles are similar, and take the general form of this: ")
st.code(body=cycle_ToSTR)
st.write("with num_epcohs=20 and batch_size=60.")
