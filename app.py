import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    loaded_model = tf.keras.models.load_model("./tomato_disease.h5")
    return loaded_model

def predict(image,loaded_model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image,[180,180])
    image = np.expand_dims(image, axis=0)
    prediction = loaded_model.predict(image)
    return prediction

model = load_model()
st.title('Tomato Disease Classification')

file = st.file_uploader('Upload an image', type=['jpg','png','jpeg'])

if file is None:
    st.text('Please upload an image file')
else:
    slot = st.empty()
    slot.text('Running inference....')

    test_image = Image.open(file)
    st.image(test_image, use_column_width=True)

    test_image = np.array(test_image)
    prediction = predict(test_image, model)
    class_names = ['Tomato_Tomato_mosaic_virus','Tomato_Tomato_YellowLeaf_Curl_virus','Tomato_Bacterial_spot','Tomato_healthy','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite']
    result = class_names[np.argmax(prediction)]
    
    output = 'The image is a ' + result
    slot.text('Done')
    st.success(output)