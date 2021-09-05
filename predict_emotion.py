# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import os
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from PIL import Image
import keras


# %%
from homomorphicfilter import HomomorphicFilter


# %%
homomorphicfilter = HomomorphicFilter(a = 0.75, b = 1.25)


# %%
model_path = "models/emotion_model.hdf5"


# %%
model = keras.models.load_model(model_path)


# %%
label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
label_indices = list(range(0, 7))
id_name_map = dict(zip(label_indices, label_names))


# %%
def filterd_and_normalized(img, homomorphicfilter):
    img_filtered = homomorphicfilter.filter(I=img, filter_params=[30,2])
    img_normalized = cv2.equalizeHist(img_filtered)
    return img_normalized


# %%
def load_image_as_grayscale_and_apply_filter_and_normalization(img_path_in, homomorphicfilter):
    img = Image.open(img_path_in)
    img = img.convert("L")
    img = np.array(img)
    img_normalized = filterd_and_normalized(img, homomorphicfilter)
    return img_normalized


# %%
def reshape_for_model(img, dim):
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized.reshape(1, *(resized.shape), 1)


# %%
def predict_image(img_path_in, homomorphicfilter, model, id_name_map, dim):
    filteredandnormalized = load_image_as_grayscale_and_apply_filter_and_normalization(img_path_in, homomorphicfilter)
    model_in = reshape_for_model(filteredandnormalized, dim)
    preds = model.predict(model_in)
    max_ind = np.argmax(preds)
    return (max_ind, id_name_map[max_ind])


# %%
def test_model_accuracy(test_image_folder):
    path, dirs, files = next(os.walk(test_image_folder))
    results = np.ones((len(files),))
    for i, file in enumerate(files):
        img_path_in = os.path.join(path, file)
        max_ind, id_name_map[max_ind] = predict_image(img_path_in, homomorphicfilter, model, id_name_map, dim=(64, 64))
        results[i] = max_ind
    return results


# %%
label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
label_indices = list(range(0, 7))
id_name_map = dict(zip(label_indices, label_names))


# %%
emotion_index = 0
emotion_name = id_name_map[emotion_index]
test_image_folder = f"archive/test/{emotion_name.lower()}"


# %%
print(emotion_name)


# %%
results = test_model_accuracy(test_image_folder)
accuracy = (results == emotion_index).sum()/(results.shape)


# %%
print(f"Accuracy for {emotion_name} = {int(10000*accuracy[0])/100.0}%")

# %% [markdown]
# # Test all labels

# %%
accuracy_dict = {}


# %%

for i, label_name in enumerate(label_names):
    test_image_folder = f"archive/test/{label_name.lower()}"
    results = test_model_accuracy(test_image_folder)
    accuracy = (results == i).sum()/(results.shape)
    accuracy_dict[label_name] = int(10000*accuracy[0])/100.0
    print(f"Completed the label {label_name}")


# %%
pprint(accuracy_dict)


