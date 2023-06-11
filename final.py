from PIL import Image, ImageDraw, ImageFont  # opening and manipulating images
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras_ocr
import math
from PIL import ImageTk
import tkinter as tk
import requests
import uuid


#TEXT EXTRACTION
pipeline = keras_ocr.pipeline.Pipeline()

images = [
    keras_ocr.tools.read(img) for img in ['D:\OCR_P1\photo2.png']
]
prediction_groups = pipeline.recognize(images)

text_dict = {}
predicted_image_1 = prediction_groups[0]

for text, box in predicted_image_1:

    text_dict[text] = box[0]

#REMOVING TEXT FROM THE BG
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

#Main function that detects text and inpaints. 
#Inputs are the image path and kreas_ocr pipeline
def inpaint_text(img_path, pipeline):
    # read the image 
    img = keras_ocr.tools.read(img_path) 
    
    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples. 
    prediction_groups = pipeline.recognize([img])
    
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
        
        #For the line thickness, we will calculate the length of the line between 
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(inpainted_img)
import requests
import uuid
#translator
def translator(word, source):
    key = "7f2623d635f944d7969e462d54f8c5cc"
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = "centralindia"
    path = '/translate'
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': source
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{'text': word}]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    translated_word = response[0]["translations"][0]["text"]

    return translated_word

pipeline = keras_ocr.pipeline.Pipeline()

img_text_removed = inpaint_text("D:\OCR_P1\photo2.png", pipeline)
image_change= cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB)
image_pil= Image.fromarray(image_change)

# Perform any necessary operations on the PIL image
# ...

# Save the modified PIL image to your system
image_pil.save("D:\OCR_P1\output_1.png")

#SEGREGATION OF WORDS ACCORDING TO THEIR Y COORDINATE
segregated_words = []

sorted_items = sorted(text_dict.items(), key=lambda x: x[1][1])

current_bottom = sorted_items[0][1][1]
current_group = []
current_coordinate = sorted_items[0][1]

# Iterate over the sorted items
for word, (left, bottom) in sorted_items:
    # Check if the bottom coordinate belongs to a new group
    if bottom != current_bottom:
        # Add the current group with the coordinate and sentence to the segregated_words list
        segregated_words.append((current_coordinate, ' '.join(current_group)))
        # Start a new group
        current_group = []
        current_bottom = bottom
        current_coordinate = (left, bottom)
    
    # Add the word to the current group
    current_group.append(word)

# Add the last group with the coordinate and sentence to the segregated_words list
segregated_words.append((current_coordinate, ' '.join(current_group)))



#Print the segregated words with coordinates and sentences
#for coordinate, sentence in segregated_words:
 #   print(f"Coordinate: {coordinate}, Sentence: {sentence}")
#output image
#image = cv2.imread("C:\\Users\\LENOVO\\Desktop\\output.png")

#r_image= cv2.imread("C:\\Users\\LENOVO\\Desktop\\photo2.png")

# Resize the image
#resized_image = cv2.resize(image, (width, height))

# define font face
fn="D:\OCR_P1\output_1.png"
img = Image.open(fn)
 
I1 = ImageDraw.Draw(img)
myFont = ImageFont.truetype('NirmalaB.ttf', 15, layout_engine=ImageFont.Layout.RAQM)
for coordinate, sentence in segregated_words:
    print(f"Coordinate: {coordinate}, Sentence: {sentence}")
L1=[]      
for list in segregated_words:
    for value,word in segregated_words:
        text=str(word)
        coordinate=tuple(map(int,value))
        L1.append((translator(text,'hi'),coordinate))
print(L1)


    



# Remove the file extension from fn
filename = os.path.splitext(fn)[0]
img.save('geeks.png')

window = tk.Tk()
window.title("Image Display")

# Open the saved image
image = Image.open('geeks.png')

# Create a Tkinter-compatible photo image from the PIL image
tk_image = ImageTk.PhotoImage(image)

# Create a label and display the image on it
label = tk.Label(window, image=tk_image)
label.pack()

# Start the Tkinter event loop
window.mainloop()
font = cv2.FONT_HERSHEY_DUPLEX


# font color, scale and thickness
color = (0, 0, 255)
font_scale = 0.4
thickness = 1

for key, value in text_dict.items():
    text= key
    print(text)
    left= value[0]
    print(left)
    bottom= value[1]
    print(bottom)
    org = (int(left), int(bottom))
    img_text_removed = cv2.putText(img_text_removed, text, org, font, font_scale, color, thickness)
    
image = img_text_removed # Replace "image.jpg" with the path to your image file
window = tk.Tk()
window.title("Image Display")
image = image.convert("RGB")
# Create a Tkinter-compatible photo image from the PIL image
tk_image = ImageTk.PhotoImage(image)

# Create a label and display the image on it
label = tk.Label(window, image=tk_image)

label.pack()

# Start the Tkinter event loop
window.mainloop()
