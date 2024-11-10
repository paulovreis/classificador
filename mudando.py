import cv2

def carregar_imagem():
    imagem = cv2.imread("urso.jpg")
    if imagem is None:
        print("Erro ao carregar a imagem.")
        exit()
    return imagem

def distorcao_quadriculado(imagem):
    """Aplica padrão quadriculado alternado (amarelo e verde)."""
    for y in range(0, imagem.shape[1], 20):
        for x in range(0, imagem.shape[0], 20):
            if (x // 20) % 2 == 0:
                imagem[x:x+10, y:y+10] = (0, 255, 255)  # Amarelo
            else:
                imagem[x:x+10, y:y+10] = (0, 255, 0)  # Verde
    return imagem

def distorcao_alterar_cores(imagem):
    """Aumenta o canal vermelho em 50 unidades."""
    imagem[:, :, 2] = cv2.add(imagem[:, :, 2], 50)
    return imagem

def distorcao_redimensionamento(imagem):
    """Redimensiona a imagem para 50% do tamanho original."""
    return cv2.resize(imagem, (0, 0), fx=0.5, fy=0.5)

def distorcao_retangulo(imagem):
    """Desenha um retângulo vermelho ao redor da imagem."""
    cv2.rectangle(imagem, (10, 10), (imagem.shape[1] - 10, imagem.shape[0] - 10), (0, 0, 255), thickness=400)
    return imagem

def distorcao_filtro_gaussiano(imagem):
    """Aplica filtro de desfoque gaussiano."""
    return cv2.GaussianBlur(imagem, (15, 15), 0)

def exibir_imagem(titulo, imagem):
    """Exibe a imagem com o título especificado."""
    cv2.imshow(titulo, imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("Escolha a distorção que deseja aplicar:")
    print("1 - Padrão Quadriculado")
    print("2 - Alteração de Cores (Canal Vermelho)")
    print("3 - Redimensionamento")
    print("4 - Retângulo Vermelho")
    print("5 - Filtro Gaussiano")
    
    escolha = input("Digite o número da distorção: ")
    imagem = carregar_imagem()

    if escolha == "1":
        imagem_distorcida = distorcao_quadriculado(imagem)
        exibir_imagem("Padrão Quadriculado", imagem_distorcida)
    elif escolha == "2":
        imagem_distorcida = distorcao_alterar_cores(imagem)
        exibir_imagem("Alteração de Cores", imagem_distorcida)
    elif escolha == "3":
        imagem_distorcida = distorcao_redimensionamento(imagem)
        exibir_imagem("Redimensionamento", imagem_distorcida)
    elif escolha == "4":
        imagem_distorcida = distorcao_retangulo(imagem)
        exibir_imagem("Retângulo Vermelho", imagem_distorcida)
    elif escolha == "5":
        imagem_distorcida = distorcao_filtro_gaussiano(imagem)
        exibir_imagem("Filtro Gaussiano", imagem_distorcida)
    else:
        print("Escolha inválida.")

if __name__ == "__main__":
    main()
