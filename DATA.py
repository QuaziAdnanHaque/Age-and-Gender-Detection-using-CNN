# importing necessary libraries 
import matplotlib.pyplot as plt 
import pandas as pd
import tensorflow as tf 
import random 
import numpy as np
import keras 
from keras import layers 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import EarlyStopping 
# reading the dataset 
df= pd.read_csv('age_gender.csv') 
print(df.head())
# information about the dataset 
df.info()
# creating a fuctio which will change the datatype to an array 
def str_to_array(ob): 
	return np.array(ob.split(' '), dtype='int') 

# apply the function to pixels column and create a new column 'new_pixels' 
df['new_pixels'] = df['pixels'].apply(str_to_array)
# creating an empty 3x3 subplot 
fig, ax = plt.subplots(3, 5, figsize=(12, 7)) 
ax = ax.ravel() 

# generating a random list of integers 
res = random.sample(range(0, df.shape[0]), 15) 

# creating the subplot for random images and printing the corresponding gender and age 
for i, id in enumerate(res): 
	ax[i].imshow(df['new_pixels'].loc[id].reshape(48, 48)) 
	ax[i].set_title(f'Age:{df.age.loc[id]}, Gender:{df.gender.loc[id]}') 
	ax[i].axis('off') 
plt.savefig('image_visualization_subplot.png')
# create the X (predictor) variables 
X_new = pd.DataFrame(df['new_pixels'].tolist()) 

# Assign Predictor Variables and Target variables 
X = X_new 
y_age = df['age'].values 
y_gender = df['gender'].values 

# split the Variables into Train and Validation sets 
y_reg_train, y_reg_val, y_clf_train, y_clf_val, X_train, X_val = train_test_split(y_age, y_gender, X, test_size=0.2, stratify = y_gender, random_state=42) 
# Check the shape 
y_reg_train.shape, y_reg_val.shape, y_clf_train.shape, y_clf_val.shape, X_train.shape, X_val.shape
# normalising the Pixel data for training dataset and then reshaping it to (48,48,1) 
Xmin = 0
Xmax = 255
X_train = X_train.values 
X_train = X_train - Xmin/(Xmax-Xmin) 
X_train = X_train.reshape(-1,48,48,1) 

# similar step is taken for Validation set 
X_val = X_val.values 
X_val = X_val - Xmin/(Xmax-Xmin) 
X_val = X_val.reshape(-1,48,48,1)
input_layer = keras.Input(shape=(48, 48, 1), name="Input image") 
x = layers.Conv2D(16, 3, activation="relu")(input_layer) 
x = layers.Conv2D(32, 3, activation="relu")(x) 
x = layers.MaxPooling2D(3)(x) 
x = layers.Conv2D(64, 3, activation="relu")(x) 
x = layers.Conv2D(64, 3, activation="relu")(x) 
x = layers.Flatten()(x) 
x = layers.Dense(128, activation='relu')(x) 
x = layers.Dense(32, activation='relu')(x) 



out_a = keras.layers.Dense(1, activation='sigmoid', name='g_clf')(x) 
out_b = keras.layers.Dense(1, activation='linear', name='a_reg')(x) 

model = keras.Model( inputs = input_layer, outputs = [out_a, out_b], name="age_gender_model")
# compile the model 
model.compile( 
	loss = { 
		"g_clf": 'binary_crossentropy', 
		"a_reg": 'mse'
	}, 

	metrics = { 
		"g_clf": 'accuracy', 
		"a_reg": 'mse'
	}, 

	optimizer = tf.keras.optimizers.Adam(learning_rate=0.003) 
) 
# create an EarlyStopping instance which will stop if the val_loss doesn't change much in 25 epochs 
callback = EarlyStopping(monitor='val_loss', patience=25, verbose=0) 
# train the model 
history = model.fit(X_train, [y_clf_train, y_reg_train], batch_size = 256, validation_data= (X_val, [y_clf_val, y_reg_val]), epochs=200, callbacks = [callback])
# plotting for Gender Classification Accuracy 
plt.plot(history.history['g_clf_accuracy'], label = 'training accuracy') 
plt.plot(history.history['val_g_clf_accuracy'], label = 'validation accuracy') 
plt.title('Gender classification model Accuracy') 
plt.xlabel('epoch') 
plt.ylabel('Accuracy for gender classification') 
plt.legend() 
plt.show()
# plotting the mse loss for age regression 
plt.plot(history.history['a_reg_mse'], label = 'training loss') 
plt.plot(history.history['val_a_reg_mse'], label = 'validation loss') 
plt.title('Age Estimation model loss') 
plt.xlabel('epoch') 
plt.ylabel('loss for age estimation') 
plt.legend() 
plt.show() 
fig, ax = plt.subplots(3,3, figsize = (10,15)) 

ax = ax.ravel() 

res = random.sample(range(0, X_val.shape[0]), 9) 

for i,id in enumerate(res): 
	ax[i].imshow(X_val[id]) 
	ax[i].set_title(f'Age-group: {y_reg_val[id]}, Gender: {gender_dict[str(y_clf_val[id])]}') 
	
	pred_Gender, pred_age = model.predict(tf.expand_dims(X_val[id], 0), verbose = 0) 
	y_value = np.where(pred_Gender > 0.5, 1,0) 
	ax[i].set_xlabel(f'gender: {gender_dict[str(y_value[0][0])]} , age: {int(np.round(pred_age,0))}') 
	
plt.savefig('prediction_subplot.png')
