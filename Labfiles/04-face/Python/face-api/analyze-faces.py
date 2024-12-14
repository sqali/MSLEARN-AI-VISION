from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Import namespaces
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials

def main():

    global face_client

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Authenticate Face client
        credential = CognitiveServicesCredentials(ai_key)
        face_client = FaceClient(ai_endpoint, credential)

        # Menu for face functions
        print('1: Detect faces\nAny other key to quit')
        command = input('Enter a number:')
        if command == '1':
            DetectFaces(os.path.join('images','people.jpg'))

    except Exception as ex:
        print(ex)

def DetectFaces(image_file):
    print('Detecting faces in', image_file)

    # Specify facial features to be retrieved
    features = [FaceAttributeType.occlusion,
                FaceAttributeType.blur,
                FaceAttributeType.glasses]
    
    # Get faces
    with open(image_file, mode = 'rb') as image_data:
        detected_faces = face_client.face.detect_with_stream(image=image_data, return_face_attributes=features, return_face_id=False)

        if len(detected_faces) > 0:
            print(len(detected_faces), 'face detected.')

            # Prepare the image for drawing
            fig = plt.figure(figsize=(8,6))
            plt.axis('off')
            image = Image.open(image_file)
            draw = ImageDraw.Draw(image)
            color = 'lightgreen'
            face_count = 0 

            # Draw and annotate each face
            for face in detected_faces:

                # Get Face Properties
                face_count += 1
                print('\nFace Number {}'.format(face_count))

                detected_attributes = face.face_attributes.as_dict()
                if 'blur' in detected_attributes:
                    print(' -blur:')
                    for blur_name in detected_attributes['blur']:
                        print(' - {}:{}'.format(blur_name, detected_attributes['blur'][blur_name]))

                if 'occlusion' in detected_attributes:
                    print(' - Occlusion')
                    for occlusion_name in detected_attributes['occlusion']:
                        print(' - {}:{}'.format(occlusion_name, detected_attributes['occlusion'][occlusion_name]))

                if 'glasses' in detected_attributes:
                    print(' - Glasses:{}'.format(detected_attributes['glasses']))

                # Draw and Annotate Faces
                r = face.face_rectangle
                bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
                draw = ImageDraw.Draw(image)
                draw.rectangle(bounding_box, outline = color, width = 5)
                annotation = 'Face ID: {}'.format(face.face_id)
                plt.annotate(annotation, (r.left, r.top), backgroundcolor=color)

            # Save Annotated Image
            plt.imshow(image)
            outputfile = 'detected_faces.jpg'
            fig.savefig(outputfile)

            print('\nResults saved in'. outputfile)
            
if __name__ == "__main__":
    main()