from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
import numpy as np

# import namespaces
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials


def main():
    global cv_client

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Get image
        image_file = 'images/people.jpg'

        # Authenticate Azure AI Vision client
        credential = CognitiveServicesCredentials(ai_key)
        cv_client = ComputerVisionClient(ai_endpoint, credential)
        
        # Analyze image
        AnalyzeFaces(image_file)

    except Exception as ex:
        print(ex)


def AnalyzeFaces(image_file):
    print('\nAnalyzing ', image_file)

    # Specify the features to be retrived (faces)
    features = [VisualFeatureTypes.faces]

    # Get Image Analysis
    with open(image_file, mode='rb') as image_data:

        analysis = cv_client.analyze_image_in_stream(image_data, features)

        # Get Faces
        if analysis.faces:
            print(len(analysis.faces), 'faces detected.')

            # Prepare the image for drawing
            fig = plt.figure(figsize=(8,6))
            plt.axis('off')
            image = Image.open(image_file)
            draw = ImageDraw.Draw(image)
            color = 'lightgreen'

            # Draw and Annotate each face
            for face in analysis.faces:
                r = face.face_rectangle
                bounding_box = ((r.left, r.right), (r.left + r.width, r.top + r.height))
                draw = ImageDraw.Draw(image)
                draw.rectangle(bounding_box, outline = color, width = 5)
                annotation = 'Person at approximately {}, {}'.format(r.left, r.top)
                plt.annotate(annotation,(r.left, r.top), backgroundcolor = color)

            # Save Annotated Image
            plt.imshow(image)
            outputfile = 'detected_faces.jpg'
            fig.savefig(outputfile)

            print("Result saved in", outputfile)

if __name__ == "__main__":
    main()