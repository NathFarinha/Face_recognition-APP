'''
    Código responsável por cadastrar novos usuários e suas fotos.
    Acrescentando-as na pasta 'Dataset'
'''

# Importando bibliotecas e arquivos necessários
from Alinhando_faces import FaceAligner
from helpers import rect_to_bb
import imutils
import dlib
import cv2
from imutils import paths
import os
import uuid

# Caminho para a pasta do Dataset com as fotos dos usuários
dataset = 'Dataset'
# Lista os endereços de localização de todas as imagens conhecidas
imagePaths = list(paths.list_images(dataset))

# Inicializa o detector de rosto do dlib (baseado em HOG) e depois cria
# o preditor de landmarks e o alinhamento' facial
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=300)

# Armazena o nome do novo usuário
username = str(input('Insira seu nome de usuário: '))
user_path = os.path.join(dataset, username)

# Informa que o nome de usuário já existe
if os.path.exists(user_path):
    print('Usuário já existe. Tente novamente.')

# Se o nome não existe, cria uma pasta com o nome de usuário cadasttrado
else:
    os.makedirs(user_path)
    print('[INFO] Pressione a tecla A no teclado para capturar nova imagem para o dataset.')
    print('[INFO] Pressione a tecla Q para sair.')
    # Estabelece uma conexão com a webcam
    cap = cv2.VideoCapture(0)
    # Enquanto a câmera estiver aberta, executa as seguintes operações:
    while cap.isOpened():
        ret, frame = cap.read()

        # Se a tecla 'A' for acionada, armazena uma captura do frame
        # na pasta referente ao novo usuário cadastrado
        if cv2.waitKey(1) & 0XFF == ord('a'):
            # Cria um caminho de arquivo exclusivo para cada imagem captada
            imgname = os.path.join(user_path, '{}.jpg'.format(uuid.uuid1()))
            image = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Nesse caso só detecção de uma pessoa
            rects = detector(gray, 2)

            # Extrai o ROI (região de interesse) da face *original*,
            # depois alinha a face usando pontos de referência faciais (landmarks)

            # Loop sobre as detecções de rosto
            for rect in rects:
                (x, y, w, h) = rect_to_bb(rect)
                faceOrig = imutils.resize(image, width=300)
                faceAligned = fa.align(image, gray, rect)

                # Armazena a imagem já com a face alinhada,
                # para facilitar o reconhecimento facial posteriormente
                cv2.imwrite(imgname, faceAligned)
                print('Imagem capturada!')

        # Mostrar imagem de volta à tela
        cv2.imshow('Imagem', frame)

        # A tecla 'Q' pausa a execução do código
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    # Fecha webcam
    cap.release()
    cv2.destroyAllWindows()









