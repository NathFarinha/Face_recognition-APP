
'''
    Código utilizado para alinhar as faces presentes em cada imagem capturada no
    código 'Capturar_imagens_dataset.py', antes de inserí-las na pasta de Dataset.
'''

# Importando bibliotecas e arquivos necessários
from helpers import FACIAL_LANDMARKS_IDXS
from helpers import shape_to_np
import numpy as np
import cv2


# Inicializando classe FaceAligner para alinhar fotografias com faces
class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=300, desiredFaceHeight=None):
        # Armazena o 'facial landmark predictor', saída desejada para a posição do olho esquerdo
        # e largura da face de saída desejada + altura
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # Se a altura da face desejada for Nenhuma (None), definimos como a
        # largura de face desejada (comportamento normal)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth


    def align(self, image, gray, rect):
        # Converte as coordenadas das landmarks (x, y) em um array NumPy
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # Extrai as coordenadas do olho esquerdo e direito (x, y)
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]


        # Calcula o centro de massa de cada olho
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # Calcula o ângulo entre os centróides dos olhos
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # Calcula a coordenada X desejada do olho direito com base na
        # coordenada X desejada do olho esquerdo
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # Determina a escala da nova imagem resultante, a partir
        # da razão da distância entre os olhos na
        # imagem *atual*, para a relação da distância entre os olhos na
        # imagem *desejada*
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist


        # Calcula as coordenadas do centro (x, y) (ou seja, o ponto mediano)
        # entre os dois olhos na imagem de entrada
        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2), int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))

        # Cria a matriz de rotação para girar e dimensionar a face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # Atualiza o componente de transação da matriz
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])


        # Aplica a transformação afim
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w+50, h+50), flags=cv2.INTER_CUBIC)

        # Retorna a face alinhada
        return output







