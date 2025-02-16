import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

#Učitavanje modela
MODEL_PATH = "model_vozila_v2.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model uspješno učitan.")
except Exception as e:
    print(f"⚠️ Greška prilikom učitavanja modela: {e}")
    exit()

class_labels = ["Big Truck", "City Car", "Multi Purpose Vehicle", "Sedan", "Sport Utility Vehicle", "Truck", "Van"]

#Funkcija za predikciju s ispravnim formatom brojeva
def predict_vehicle(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    response = {class_labels[i]: round(predictions[i], 4) for i in range(len(class_labels))}

    return response

#Gradio sučelje
interface = gr.Interface(
    fn=predict_vehicle,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=7),
    title="Klasifikacija Vozila",
    description="Uploadaj sliku vozila i dobit ćeš predikcije."
)

if __name__ == "__main__":
    interface.launch(share=True)