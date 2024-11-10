import cv2

print("Iniciando o opencv")

imagem = cv2.imread("pablo2.jpg")

cv2.imshow("ImagemOriginal", imagem)

proporcao = 100.0 / imagem.shape[1]
tamanho_novo = (100, int(imagem.shape[0] * proporcao))
img_redimensionada = cv2.resize(imagem, tamanho_novo, interpolation = cv2.INTER_AREA)
cv2.imshow("ImagemRedimensionada", img_redimensionada)

cv2.waitKey(0)