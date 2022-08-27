'''
    Código principal, responsável pelas funções do aplicativo.
    Conectado com o arquivo 'index.html'
'''

# Importando bibliotecas e arquivos necessários
from flask import Flask, render_template, request, Response
import cv2
import time
import os
import numpy as np
from imutils import paths
import face_recognition
import imutils
import dlib
from Alinhando_faces import FaceAligner
from imutils import face_utils

# Cria variáveis globais utilizadas em várias funções deste código
global grey, switch, neg, face, fr, land

grey = 0
neg = 0
face = 0
switch = 1
fr = 0
land = 0

# Caminho para a pasta do Dataset com as fotos dos usuários
dataset = 'Dataset'
app = Flask(__name__)

# Lista os endereços de localização de todas as imagens conhecidas
imagePaths = list(paths.list_images(dataset))
process_this_frame = True

# Cria algumas variáveis úteis no andamento do código
face_names = []
knownEncodings = []
knownNames = []
# Carrega o modelo de detecção de rosto pré-treinado
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt',
                               'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Variáveis auxiliares para a função visualize_landmarks()
# detectando face, alinhando a face e prevendo as landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=300)


# Inicializar aplicativo Flask
app = Flask(__name__, template_folder='./templates')
# Inicializa a câmera
camera = cv2.VideoCapture(0)

# Loop nas imagens conhecidas do Dataset para obter informações das fotos
# para que o computador possa analisar
for (i, imagePath) in enumerate(imagePaths):
    # Extrai o nome da pessoa do caminho da imagem
    name = imagePath.split(os.path.sep)[-2]
    # Carrega a imagem de entrada e faz a conversão de BGR (ordenação OpenCV)
    # para ordenação dlib (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detecta as coordenadas (x, y) das caixas delimitadoras
    # correspondentes a cada face na imagem de entrada
    boxes = face_recognition.face_locations(rgb, model='hog')
    # Calcula a codificação facial para o rosto
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
        # Adiciona cada codificação + nome ao nosso conjunto de nomes e codificações conhecidos
        knownEncodings.append(encoding)
        knownNames.append(name)


# Função para detectar apenas a face em cada frame
def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    # Retorna um blob que a imagem de entrada
    # após a subtração média, normalização e troca de canal.
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    # Armazena as detecções e confianças
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    # Se a confiança de detecção for maior que 50%,
    # a função retorna o frame
    if confidence < 0.5:
        return frame

    # Coordenadas de cada vértice da detecção (formato de retângulo)
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    box_coord = (startX, startY, endX, endY)

    # Tenta executar o novo dimensionamento do frame,
    # retornando o frame recortado apenas com a face
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    # Se não for possível executar, pula a etapa e retorna o frame inicial
    except Exception as e:
        pass
    return frame

# Função para visualizar os pontos principais da face (landmarks) em cada frame
def visualize_landmarks(frame):
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detectar faces na imagem em cinza
    rects = detector(gray, 1)

    # Loop sobre as detecções de rosto
    for (i, rect) in enumerate(rects):
        # Determina as landmarks para a região do rosto e, em seguida,
        # converte as coordenadas desses landmarks (x, y) para um NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # Converte o retângulo do dlib em uma caixa delimitadora no estilo OpenCV
        # [ou seja, (x, y, w, h)], depois desenha a caixa delimitadora da face
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        # Faz um loop sobre as coordenadas (x, y) para os landmarks
        # e os desenha na imagem
        for (x, y) in shape:
            frame = cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    # Retorna a imagem de saída com as detecções de face +
    # pontos de referência faciais (landmarks desenhadas)
    return frame

# Função para reconhecimento facial em cada frame
def face_recog(frame):
    # Converte o frame de entrada de BGR para RGB e redimensiona-o para ter
    # uma largura de 750px (para acelerar o processamento)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=500)
    r = frame.shape[1] / float(rgb.shape[1])

    # Detecta as coordenadas (x, y) das caixas delimitadoras
    # correspondente a cada face no quadro de entrada, então calcula
    # as codificações faciais para cada rosto,
    # agora utilizando a biblioteca 'face_recognition'
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Loop sobre as codificações de todas as faces detectadas
    for encoding in encodings:
        # Tenta corresponder cada face na imagem de entrada
        # com as codificações (encodings) conhecidas
        matches = face_recognition.compare_faces(knownEncodings, encoding)
        name = "Desconhecido"

        # Verifica se há uma correspondência
        if True in matches:
            # Encontra os índices de todas as faces correspondentes e inicializa um
            # dicionário para contar o número total de vezes que cada face
            # foi correspondido
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Faz um loop sobre os índices correspondentes e mantém uma contagem para
            # cada rosto reconhecido
            for i in matchedIdxs:
                name = knownNames[i]
                counts[name] = counts.get(name, 0) + 1

            # Determina o rosto reconhecido com o maior número
            # de votos (nota: no caso de um empate improvável, o Python
            # irá selecionar a primeira entrada no dicionário)
            name = max(counts, key=counts.get)

        # Atualiza a lista de nomes
        names.append(name)

    # Loop sobre os rostos reconhecidos
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Redimensiona as coordenadas da face
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # Desenha o nome do rosto previsto na imagem e o retângulo de detecção
        frame = cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        frame = cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    # Retorna o frame com o retângulo e o nome do usuário desenhados
    return frame

# Função de exibição de cada ferramenta do site, gera frame por frame da câmera
def gen_frames():
    global fr, land, face
    # Loop para todos os frames capturados do vídeo da webcam em tempo real
    while True:
        success, frame = camera.read()
        # Se a leitura do frame for bem sucedida,
        # executa o funcionamento de determinado botão acionado no site
        if success:
            # Com o botão 'Face' acionado, vemos apenas a face localizada no vídeo resultante
            if (face):
                frame = detect_face(frame)

            # Com o botão 'Grey' acionado, vemos as imagens na cor cinza no vídeo resultante
            if (grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Com o botão 'Negative' acionado, vemos as imagens
            # depois das operações bitwise no vídeo resultante
            if (neg):
                frame = cv2.bitwise_not(frame)

            # Com o botão 'Landmarks' acionado, vemos todos os
            # 68 pontos principais da face, desenhado no vídeo resultante
            if (land):
                frame = visualize_landmarks(frame)

            # Com o botão 'Face Recognition' acionado, vemos um retângulo ao redor da face
            # e o nome da pessoa reconhecida (se não estiver cadastrada, aperece 'Desconhecido')
            if (fr):
                frame = face_recog(frame)

            # Tenta exibir os frames, mostrando os resultados em tempo real o site
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # Se não conseguir ou nenhum botão for acionado, pula a etapa
            except Exception as e:
                pass
        # Se o frame não conseguir ser lido, pula a etapa
        else:
            pass


# Criando os links do site
# Primeiro a Home page, que mostra todos os botões do site.
# A função render_template() conecta o código em Python com o arquivo html,
# isso possibilita verificar se os botões foram acionados ou não,
# chamando as determinadas funções neste código python
@app.route('/')
def index():
    return render_template('index.html')

# Link para exibir o vídeo da webcam em tempo real com os resultados
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Link para chamada das funções que cada botão é responsável
@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg

        elif request.form.get('land') == 'Landmarks':
            global land
            land = not land

        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('fr') == 'Face Recognition':
            global fr
            fr = not fr

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


# Inicializando o App, inserindo uma senha padrão
if __name__ == '__main__':
    app.secret_key = "nathaliafarinha"
    app.run()





