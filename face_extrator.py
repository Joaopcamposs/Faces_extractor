import cv2
import os
import numpy as np

# Parametros para reconhecimento facial
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
width, height = 750, 750

# Diretorio raiz
dir = os.getcwd()
# Diretorio dos videos
dir_videos = f'{os.getcwd()}\\videos'
videos = []


def faces_extractor(video_path, person, ratio):
    # Obter video a partir do caminho
    video = cv2.VideoCapture(video_path)

    # Flags de variaveis
    count = 0
    img_count = 1
    success = 1
    limit = 50
    ratio = ratio  # padrao 7, video 5s
    try:
        while success:
            success, frame = video.read()

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = classifier.detectMultiScale(gray_image, scaleFactor=1.5, minSize=(150, 150))

            for (x, y, l, a) in detected_faces:
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
                if np.average(gray_image) > 110:  # captura apenas se a media de luminosidade for maior que 110
                    if count % ratio == 0 and img_count <= limit:
                        face_image = gray_image[y:y + a, x:x + l]
                        cv2.imwrite(f'{dir}\\face_images\\pessoa.{person}.{img_count}.jpg', face_image)
                        img_count += 1
                count += 1
    except Exception as err:
        print(err)


# Funcao para capturar face de todos os videos da lista videos
def all_faces_capture():
    person_count = 1
    for filename in videos:
        faces_extractor(f'{dir_videos}\\{filename}', person_count, 7)
        person_count += 1


# Percorrer pasta e adicionar nome dos arquivos na lista
for (dirpath, dirnames, filenames) in os.walk(dir_videos):
    videos.extend(filenames)
# print(videos)

# Capturar faces de todos os videos da pasta
# all_faces_capture()

# Capturar faces de video especifico
# faces_extractor(f'{dir_videos}\\robson_junior.mp4', person=20, ratio=8)


