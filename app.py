import streamlit as st
import tensorflow as tf
# import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource()
def load_model():
  model=tf.keras.models.load_model('model/VGG16.h5')
  # model=tf.keras.models.load_model('model/model_eff_95train_92test.keras')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Weather Classification
         """
         )

file = st.file_uploader("Please upload an weather file", type=["jpg", "png","jpeg"])



def import_and_predict(image_data, model):
    
        size = (224, 224)   
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image)
        print('#' * 20 , image.shape)
        tmp = np.zeros_like(image)
        tmp[..., 0] = image[..., 2]
        tmp[..., 1] = image[..., 1]
        tmp[..., 2] = image[..., 0]

        img = tmp
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])

    class_names=['dew','fogsmog', 'frost'  ,'glaze','hail','lightning','rain','rainbow','rime','sandstorm','snow']
    
    st.success("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
