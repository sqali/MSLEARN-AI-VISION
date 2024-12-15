from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import sys
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

def main():
    global cv_client

    try:
        # Load configuration settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Authenticate Azure AI Vision client
        cv_client = ComputerVisionClient(ai_endpoint, CognitiveServicesCredentials(ai_key))

        # Get image file
        image_file = 'images/people.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Analyze image
        AnalyzeImage(image_file, image_file, cv_client)

    except Exception as ex:
        print("Error:", ex)

def AnalyzeImage(filename, image_file, cv_client):
    print('\nAnalyzing', filename)

    # Specify the features to retrieve (e.g., objects, people)
    features = [VisualFeatureTypes.Objects]

    # Analyze the image
    with open(image_file, "rb") as image_data:
        result = cv_client.analyze_image_in_stream(image_data, visual_features=features)

    # Identify people in the image
    people_detected = [obj for obj in result.objects if obj.object_property == "person"]

    if people_detected:
        print(f"\nDetected {len(people_detected)} people in the image.")

        # Prepare the image for drawing
        image = Image.open(filename)
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        # Draw bounding boxes around detected people
        for person in people_detected:
            r = person.rectangle
            bounding_box = [(r.x, r.y), (r.x + r.w, r.y + r.h)]
            draw.rectangle(bounding_box, outline=color, width=3)

        # Save annotated image
        plt.imshow(image)
        plt.tight_layout(pad=0)
        outputfile = 'people_detected.jpg'
        fig.savefig(outputfile)
        print('Results saved in', outputfile)

    else:
        print("No people detected in the image.")

if __name__ == "__main__":
    main()