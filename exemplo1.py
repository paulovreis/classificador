import cv2

print("iniciando o openCV")

imagem = cv2.imread("pablo2.jpg")

(b, g, r) = imagem[0,0]

print("O pixel [0,0] tem as seguintes cores: ")
print(f'vermelho: {r}, verde: {g}, azul: {b}')

cv2.imshow("janela", imagem)
cv2.waitKey(0)