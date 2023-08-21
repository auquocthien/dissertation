
import mediapipe as mp
import cv2
import numpy as np
import imageio
from tensorflow_docs.vis import embed


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


mp_drawing = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

# STEP 3: Load the input image.


def to_gif(image_path, images, duration):
    """Converts image sequence (4D numpy array) to gif."""
    imageio.mimsave('./output_gif/land_{}.gif'.format(image_path),
                    images, duration=duration)
    return embed.embed_file('./output_gif/land_{}.gif'.format(image_path))


def get_video_assarray(path):
    frames = []
    cap = cv2.VideoCapture(path)
    read = True
    while read:
        read, image = cap.read()
        if read:
            frames.append(image)
    return np.stack(frames, axis=0)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent/100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


image_path = 'dataset\D0536.mp4'
image_name = image_path.split("\\")[-1].split(".")[0]
image = get_video_assarray(image_path)

print(image.shape)
# image = mp.Image.create_from_file(image_path)
ouput_image = []
num_frames, image_height, image_witdth, _ = image.shape
with mp_hand.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
    for frame_idx in range(num_frames):
        mp_image = image[frame_idx, :, :, :]

        mp_image = cv2.cvtColor(mp_image, cv2.COLOR_BGR2RGB)
        mp_image.flags.writeable = False
        # STEP 4: Detect hand landmarks from the input image.
        detection_result = hands.process(mp_image)

        mp_image.flags.writeable = True
        mp_image = cv2.cvtColor(mp_image, cv2.COLOR_RGB2BGR)

        if detection_result.multi_hand_landmarks:
            for hand_landmarks in detection_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    mp_image, hand_landmarks, mp_hand.HAND_CONNECTIONS)

        ouput_image.append(mp_image)

output = np.stack(ouput_image, axis=0)

to_gif(image_name, output, 100)
