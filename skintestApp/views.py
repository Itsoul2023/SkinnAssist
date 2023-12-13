
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from .prediction.predict import getPrediction
#from .detect_skin_disease import detect_skin_disease

# Your existing views
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .prediction.predict import getPrediction
import cv2


def create_skin_disease_and_grain_mask(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask for skin color
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply morphological operations to remove noise in skin mask
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Define the lower and upper bounds for the color of the grains in HSV
    lower_grain = np.array([0, 20, 20], dtype=np.uint8)
    upper_grain = np.array([30, 255, 255], dtype=np.uint8)

    # Create a binary mask for grains
    grain_mask = cv2.inRange(hsv_image, lower_grain, upper_grain)

    # Apply morphological operations to remove noise in grain mask
    grain_mask = cv2.morphologyEx(grain_mask, cv2.MORPH_OPEN, kernel)
    grain_mask = cv2.morphologyEx(grain_mask, cv2.MORPH_CLOSE, kernel)

    # Combine the skin and grain masks
    combined_mask = cv2.bitwise_or(skin_mask, grain_mask)

    return combined_mask

def auto_crop_skin_disease_and_grains(image, combined_mask):
    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the contour with the largest area
        contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding to the bounding box
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        # Crop the image to include the skin disease and grain region
        cropped_image = image[y:y + h, x:x + w]
        return cropped_image
    else:
        return image

def update_location(request):
    if request.method == 'POST':
        data = request.POST.get('latitude'), request.POST.get('longitude')
        print(f"Received location: Latitude - {data[0]}, Longitude - {data[1]}")
        return JsonResponse({'message': 'Location received successfully'})
    return JsonResponse({'error': 'Invalid request method'})


@csrf_exempt
def detect_skin_disease(request):
    if request.method == 'POST':
        # Assuming your image file input name is 'image'
        uploaded_image = request.FILES.get('image')

        if not uploaded_image:
            return JsonResponse({'error': 'No image file provided'})


        # Save the uploaded image to a temporary location (change as needed)
        with open('temp_image.jpg', 'wb') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)
        image_path='temp_image.jpg'
        image = cv2.imread(image_path)

# Create combined mask for skin disease and grains
        combined_mask = create_skin_disease_and_grain_mask(image)

# Auto-crop based on the combined mask
        uploaded_image = auto_crop_skin_disease_and_grains(image, combined_mask)
        cv2.imwrite('temp_image.jpg', uploaded_image)

        # Use your prediction function to get the result
        prediction_result, accuracy, graph_data = getPrediction('temp_image.jpg')
        print("1",prediction_result)
        print(accuracy)
        print(graph_data)
        
        

        # Dummy response for testing
        result = {
            'diagnosis': prediction_result,
            'confidence': accuracy,
            'graph_data': graph_data,
        }

        return JsonResponse(result)

    # Handle other HTTP methods if needed
    return JsonResponse({'error': 'Invalid request method'})

def homePage(request):
    return render(request, 'index.html')

def aboutus(request):
    return render(request, 'aboutus.html')

def check(request):
    return render(request, 'check.html')

def policy(request):
    return render(request, 'policy.html')

def terms(request):
    return render(request, 'terms.html')

def upload(request):
    return render(request, 'upload.html')

def agree(request):
    return render(request,'agree.html')

# @csrf_exempt
# def detect_skin_disease(request):
#     if request.method == 'POST':
#         # Assuming your image file input name is 'image'
#         uploaded_image = request.FILES.get('image')

#         if not uploaded_image:
#             return JsonResponse({'error': 'No image file provided'})

#         # Save the uploaded image to a temporary location (change as needed)
#         with open('temp_image.jpg', 'wb') as destination:
#             for chunk in uploaded_image.chunks():
#                 destination.write(chunk)

#         # Use your prediction function to get the result
#         prediction_result = getPrediction('temp_image.jpg')

#         # Dummy accuracy for testing
#         accuracy = 90.5

#         # Dummy response for testing
#         result = {
#             'prediction': prediction_result,
#             'accuracy': accuracy,
#         }

#         return JsonResponse(result)

#     # Handle other HTTP methods if needed
#     return JsonResponse({'error': 'Invalid request method'})



from .forms import ImageUploadForm
import cv2
import numpy as np
from matplotlib import pyplot as plt
#Open a simple image

def get_hist(image_path):
    img=cv2.imread(image_path)
    img = cv2.resize(img, (256,256))

    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)
    
    
    cv2.imwrite("temp.jpg",global_result)
    hist = cv2.calcHist([global_result],[0],None,[256],[0,256])
    
    return hist


def check_skin(image_name):
    hist = get_hist(image_name)
    a = hist[0]
    b = hist[255]
    percent = ((a / (a + b)) * 100.0).round(2)
    print("------",percent)

    if percent > 40.00:
        #print('True')
        return True
    else:
        #print('False')
        return False
        
        
def calculate_skin_percentage(mask):
    # Calculate the number of skin pixels
    skin_pixels = np.sum(mask == 255)

    # Calculate the total number of pixels in the mask
    total_pixels = mask.size

    # Calculate the percentage of skin in the image
    skin_percentage = (skin_pixels / total_pixels) * 100

    return skin_percentage

def skin_detection(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Calculate the skin percentage
    skin_percentage = calculate_skin_percentage(mask)

    # Bitwise AND the original image and the mask
    result = cv2.bitwise_and(img, img, mask=mask)
    print("----",skin_percentage)

    if skin_percentage > 40.00:
        return True
    else:
        return False
@csrf_exempt
def skin_detection_view(request):
    if request.method == 'POST':
        cimg=request.FILES.get('image')
        if cimg:
            with open('temp_image.jpg', 'wb') as destination:
                for chunk in cimg.chunks():
                    destination.write(chunk)
            is_skin={
                'skin':skin_detection('temp_image.jpg'),
            }
            return JsonResponse(is_skin)
    

