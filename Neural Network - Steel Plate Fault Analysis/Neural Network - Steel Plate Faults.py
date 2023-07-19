#!/usr/bin/env python
# coding: utf-8

# # Neural Networks - Steel Plate Fault Analysis
# ### Liam Caulfield
# ### May 28, 2023
# 
# Predict types of steel plate faults using Neural Networks. 

# In[36]:


# import necessary libraries 
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


# In[20]:


# load dataset
df = pd.read_csv('faults.csv')
df.head(1)


# In[29]:


# Separate the independent variables (attributes) and the dependent variables (fault types)
X = df.drop(['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis=1)
y = df[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']]


# In[30]:


# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[31]:


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


# Create the neural network model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='sigmoid'))  # Output layer with 7 units for the 7 fault types


# In[33]:


# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[34]:


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


# In[35]:


# Evaluate the model on the testing set
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f%%' % (accuracy * 100))


# In[48]:


epochs = range(1, 51)  # Assuming 50 epochs based on the provided training results
loss = [0.6305, 0.4851, 0.3704, 0.3109, 0.2857, 0.2688, 0.2551, 0.2435, 0.2338, 0.2256,
        0.2192, 0.2136, 0.2091, 0.2049, 0.2019, 0.1995, 0.1967, 0.1941, 0.1920, 0.1903,
        0.1882, 0.1866, 0.1854, 0.1837, 0.1820, 0.1808, 0.1792, 0.1778, 0.1767, 0.1752,
        0.1743, 0.1735, 0.1721, 0.1707, 0.1696, 0.1693, 0.1673, 0.1664, 0.1662, 0.1645,
        0.1636, 0.1625, 0.1617, 0.1608, 0.1598, 0.1596, 0.1578, 0.1580, 0.1565, 0.1553]
accuracy = [0.3175, 0.3691, 0.4408, 0.4843, 0.4956, 0.5286, 0.5447, 0.5786, 0.6245, 0.6430,
            0.6640, 0.6712, 0.6849, 0.6857, 0.6922, 0.6922, 0.7019, 0.7091, 0.7083, 0.7091,
            0.7147, 0.7147, 0.7196, 0.7212, 0.7268, 0.7276, 0.7397, 0.7349, 0.7405, 0.7373,
            0.7462, 0.7454, 0.7470, 0.7462, 0.7542, 0.7575, 0.7566, 0.7623, 0.7558, 0.7615,
            0.7655, 0.7655, 0.7647, 0.7663, 0.7712, 0.7647, 0.7671, 0.7703, 0.7720, 0.7776]


# In[49]:


# Loss plot
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy plot
plt.plot(epochs, accuracy, 'r', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ***Findings Summary:***
# The model was trained using a dataset consisting of 39 samples with independent variables and corresponding target values. The training was performed for 50 epochs, and the results indicate an improvement in both loss and accuracy over time.
# 
# During training, the loss decreased gradually from an initial value of 0.6305 to 0.1553, while the accuracy increased from 31.75% to 77.76%. These improvements demonstrate that the model successfully learned the patterns and relationships within the training data.
# 
# The validation results also show promising performance, with a final loss of 0.1835 and an accuracy of 70.42%. This indicates that the model generalized well to unseen data, which is important for its practical application.
# 
# In summary, the trained model exhibits good predictive capabilities with an accuracy of 72.24%. It successfully learned from the provided dataset and showed promising performance in predicting the target values based on the given independent variables.
