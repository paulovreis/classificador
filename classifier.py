#
# Depends on:
# - tensorflow: pip install tensorflow
# - keras: pip install keras
# - opencv: pip install opencv-python
# - scikit-image: pip install scikit-image
# - pandas: pip install pandas

import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import pandas as pd

# Carregar o modelo ResNet50
model = ResNet50(weights='imagenet')

# Definir o caminho para o dataset Imagenette2
imagenette_path = "C:/Users/Pichau/Documents/GitHub/classificador/imagenette2"

# Função para carregar e pré-processar a imagem
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1].astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Função para classificar uma imagem e obter a classe e probabilidade
def classify_image(img):
    x = preprocess_image(img)
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=1)[0][0]
    return decoded_preds[1], decoded_preds[2]  # Classe e certeza

# Função para calcular a similaridade estrutural (SSIM)
def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True) * 100

# Distorções definidas
def distorcao_quadriculado(imagem):
    for y in range(0, imagem.shape[1], 20):
        for x in range(0, imagem.shape[0], 20):
            if (x // 20) % 2 == 0:
                imagem[x : x + 10, y : y + 10] = (0, 255, 255)  # Amarelo
            else:
                imagem[x : x + 10, y : y + 10] = (0, 255, 0)  # Verde
    return imagem

def distorcao_alterar_cores(imagem):
    imagem[:, :, 2] = cv2.add(imagem[:, :, 2], 50)
    return imagem

def distorcao_redimensionamento(imagem):
    return cv2.resize(imagem, (0, 0), fx=0.5, fy=0.5)

def distorcao_retangulo(imagem):
    cv2.rectangle(imagem, (10, 10), (imagem.shape[1] - 10, imagem.shape[0] - 10), (0, 0, 255), thickness=8)
    return imagem

def distorcao_filtro_gaussiano(imagem):
    return cv2.GaussianBlur(imagem, (15, 15), 0)

# Função para processar todas as imagens
def process_imagenette_dataset():
    data = []
    
    # Percorrer todas as imagens na pasta
    for root, dirs, files in os.walk(imagenette_path):
        for file in files:
            file_path = os.path.join(root, file)
            original_img = cv2.imread(file_path)

            # Obter classificação e certeza da imagem original
            orig_class, orig_confidence = classify_image(original_img)

            # Aplicar cada distorção, calcular SSIM e nova classificação
            for dist_name, dist_func in [
                ("Quadriculado", distorcao_quadriculado),
                ("Alterar Cores", distorcao_alterar_cores),
                ("Redimensionamento", distorcao_redimensionamento),
                ("Retângulo", distorcao_retangulo),
                ("Filtro Gaussiano", distorcao_filtro_gaussiano),
            ]:
                # Copiar a imagem original para aplicar a distorção
                distorted_img = dist_func(original_img.copy())

                # Calcular SSIM entre original e distorcida
                similarity = calculate_ssim(original_img, distorted_img)

                # Classificar imagem distorcida
                dist_class, dist_confidence = classify_image(distorted_img)

                # Verificar mudança de classe
                class_changed = orig_class != dist_class

                # Registrar dados
                data.append({
                    "Image": file,
                    "Distortion": dist_name,
                    "Original Class": orig_class,
                    "Original Confidence": orig_confidence,
                    "Distorted Class": dist_class,
                    "Distorted Confidence": dist_confidence,
                    "Class Changed": class_changed,
                    "SSIM": similarity,
                })
    
    # Salvar dados em um DataFrame para análise
    df = pd.DataFrame(data)
    df.to_csv("imagenette_distorcoes_resultados.csv", index=False)
    print("Processamento completo. Resultados salvos em 'imagenette_distorcoes_resultados.csv'.")

# Executar o processamento
process_imagenette_dataset()
