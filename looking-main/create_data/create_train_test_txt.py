import os

folder_txt = r"C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\output_txt"
os.makedirs(folder_txt, exist_ok=True)  # Cria o diretório caso não exista

for file_name in ["train.txt", "val.txt", "test.txt"]:
    open(os.path.join(folder_txt, file_name), "w").close()  # Cria arquivos vazios
