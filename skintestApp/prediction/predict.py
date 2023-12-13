# import numpy as np
# from PIL import Image
# from sklearn.preprocessing import LabelEncoder
# from keras.models import load_model

# import os

# # Get the directory of the current script
# script_dir = os.path.dirname(__file__)

# # Specify the path to your model file
# model_path = os.path.join(script_dir, 'model1.h5')

# # Load the model
# my_model = load_model(model_path)

# classes = ['Acne/Rosacea',
#            'Actinic Keratosis/Basal Cell Carcinoma/Malignant Lesions',
#            'Eczema',
#           'Melanoma Skin Cancer/Nevi/Moles',
#          'Psoriasis/Lichen Planus and related diseases', 
#            'Tinea Ringworm/Candidiasis/Fungal Infections',
#           'Urticaria/Hives', 
#            'Nail Fungus/Nail Disease']

# def getPrediction(img_path):
#     le = LabelEncoder()
#     le.fit(classes)

#     SIZE = 32  # Resize to the same size as training images

#     # Open the uploaded image and resize it
#     img = np.asarray(Image.open(img_path).resize((SIZE, SIZE)))

#     img = img / 255.  # Scale pixel values

#     img = np.expand_dims(img, axis=0)  # Get it ready as input to the network       

#     pred = my_model.predict(img)  # Predict

#     # Convert prediction to class name
#     pred_class = le.inverse_transform([np.argmax(pred)])[0]

#     confidence = pred[0][np.argmax(pred)] * 100

#     # Create a dictionary for graph data
#     graph_data = {cls: round(pred[0][i] * 100, 2) for i, cls in enumerate(classes)}
#     print("Diagnosis is:", pred_class)
#     print("Confidence: ", confidence, "%")
#     print("Graph Data: ", graph_data)

#     return pred_class, confidence, graph_data



# from keras.models import load_model
# import tensorflow as tf
# import cv2
# import os

# classes = {0:'Acne/Rosacea',
#            1:'Actinic Keratosis/Basal Cell Carcinoma/Malignant Lesions',
#            2:'Eczema',
#            3:'Melanoma Skin Cancer/Nevi/Moles',
#            4:'Psoriasis/Lichen Planus and related diseases', 
#            5:'Tinea Ringworm/Candidiasis/Fungal Infections',
#            6:'Urticaria/Hives', 
#            7:'Nail Fungus/Nail Disease'}


# #model = load_model('model.h5', compile=True)
# script_dir = os.path.dirname(__file__)

# model_path = os.path.join(script_dir, 'model1.h5')


# model = load_model(model_path)

# def getPrediction(image):
#     img = cv2.imread(image)
#     img = cv2.resize(img, (32,32)) / 255.0

#     predictions = (model.predict(img.reshape(1,32,32,3)) * 100.0).round(2)
#     new_dict = {
#         classes[0]: predictions[0][0],
#         classes[1]: predictions[0][1],
#         classes[2]: predictions[0][2],
#         classes[3]: predictions[0][3],
#         classes[4]: predictions[0][4],
#         classes[5]: predictions[0][5],
#         classes[6]: predictions[0][6],
#         classes[7]: predictions[0][7]
#     }

#     dict_dis = sorted(new_dict.items(), key=lambda x: x[1], reverse=True)
#     dict_dis = dict(sorted(new_dict.items(), key=lambda x: x[1], reverse=True)[:3])
#     graph_data=dict_dis
    
    
#     max_val = max(dict_dis, key=dict_dis.get)
  
#        # print(str(max_val),'percentage',str(dict_dis[max_val]))
#     pred_class =(str(max_val))
#     confidence=str(dict_dis[max_val])
#     # print(pred_class)
#     # print(confidence)
#     # print(graph_data)
    
#     return pred_class, confidence, graph_data

# from keras.models import load_model
# import tensorflow as tf
# import cv2
# import os

# classes = {
#     0: 'Acne/Rosacea',
#     1: 'Actinic Keratosis/Basal Cell Carcinoma/Malignant Lesions',
#     2: 'Eczema',
#     3: 'Melanoma Skin Cancer/Nevi/Moles',
#     4: 'Psoriasis/Lichen Planus and related diseases',
#     5: 'Tinea Ringworm/Candidiasis/Fungal Infections',
#     6: 'Urticaria/Hives',
#     7: 'Nail Fungus/Nail Disease'
# }

# # model = load_model('model.h5', compile=True)
# script_dir = os.path.dirname(__file__)
# model_path = os.path.join(script_dir, 'new_model1.h5')

# model = load_model(model_path)

# def getPrediction(image):
#     img = cv2.imread(image)
#     img = cv2.resize(img, (32, 32)) / 255.0

#     predictions = model.predict(img.reshape(1, 32, 32, 3)) * 100.0
#     predictions = predictions.flatten()

#     # Convert class indices to class names
#     class_names = [classes[i] for i in range(len(classes))]

#     # Create a dictionary with class names and corresponding percentages
#     result_dict = {class_name: str(predictions[i]) for i, class_name in enumerate(class_names)}

#     # Sort the dictionary by percentages in descending order
#     sorted_result = dict(sorted(result_dict.items(), key=lambda item: float(item[1]), reverse=True))

#     # Get the predicted class and confidence
#     pred_class = next(iter(sorted_result))
#     confidence = sorted_result[pred_class]

#     # Return the predicted class, confidence, and graph data
#     return pred_class, confidence, sorted_result


from keras.models import load_model
import cv2
import os
import gdown

classes = {
    0: 'Acne/Rosacea',
    1: 'Actinic Keratosis/Basal Cell Carcinoma/Malignant Lesions',
    2: 'Eczema',
    3: 'Melanoma Skin Cancer/Nevi/Moles',
    4: 'Psoriasis/Lichen Planus and related diseases',
    5: 'Tinea Ringworm/Candidiasis/Fungal Infections',
    6: 'Urticaria/Hives',
    7: 'Nail Fungus/Nail Disease'
}

# Google Drive file ID
file_id = '1gHvALQpEyIYELg6EUTMPQoO3CTre_7sA'
# Specify the destination path to save the model file
destination_path = 'fmodel.h5'

# Download the model file from Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', destination_path, quiet=False)

# Load the model
model = load_model(destination_path)

def getPrediction(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (32, 32)) / 255.0

    predictions = model.predict(img.reshape(1, 32, 32, 3)) * 100.0
    predictions = predictions.flatten()

    # Convert class indices to class names
    class_names = [classes[i] for i in range(len(classes))]

    # Create a dictionary with class names and corresponding percentages
    result_dict = {class_name: str(predictions[i]) for i, class_name in enumerate(class_names)}

    # Sort the dictionary by percentages in descending order
    sorted_result = dict(sorted(result_dict.items(), key=lambda item: float(item[1]), reverse=True))

    # Get the predicted class and confidence
    pred_class = next(iter(sorted_result))
    confidence = sorted_result[pred_class]

    # Return the predicted class, confidence, and graph data
    return pred_class, confidence, sorted_result




