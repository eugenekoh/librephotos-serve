import cv2
import torch
from loguru import logger
from skimage.feature import hog
from torchvision import transforms

from fer.models import FERModel

MODEL_PATH = './fer/ResMasking_V7.2.pth'
device = torch.device('cpu')
model = FERModel(1, 7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
classes = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def predict_emotion(img):
    logger.info("predicting emotion")

    def img2tensor(x):
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        return transform(x)

    # preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))  # resizes to 48x48 sized image

    # transform 
    img = img2tensor(resized)

    # predict the mood
    temp = torch.squeeze(img)
    fv = hog(temp, orientations=9, pixels_per_cell=(4, 4),
             cells_per_block=(1, 1), feature_vector=True)
    fv = torch.unsqueeze(torch.FloatTensor(fv.ravel()), 0)
    img = torch.unsqueeze(img, 0)
    out = model(img, fv)
    softmax = torch.nn.Softmax(dim=1)
    scaled = softmax(out)
    prob = torch.max(scaled).item()
    label = classes[torch.argmax(scaled).item()]

    result = {'emotion': label, 'confidence': prob}
    logger.info(result)

    return result
