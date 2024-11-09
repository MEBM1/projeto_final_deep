from utils import JAAD_loader, JAAD_creator, Kitti_creator, PIE_loader, PIE_creator
import os

# Definindo os caminhos diretamente
jaad_path = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\JAAD"
path_joints = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\output_pifpaf_01_2k30\\output_pifpaf_01_2k30"
txt_out = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\output_txt"
dir_out = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\pasta_aux"
data = "JAAD"

# Carregando e criando o dataset JAAD
if 'JAAD' in data:
    jaad_loader = JAAD_loader(jaad_path, txt_out)
    #dict_annotation = jaad_loader.generate()
    dict_annotation = jaad_loader.generate_with_attributes_ann()
    jaad_creator = JAAD_creator(txt_out, dir_out, path_joints, os.path.join(jaad_path, 'images'))
    jaad_creator.create(dict_annotation)

# Carregando e criando o dataset PIE (caso necessário)
if 'PIE' in data:
    pie_loader = PIE_loader(jaad_path)
    data = pie_loader.generate()

    creator = PIE_creator(txt_out, dir_out, path_joints, os.path.join(jaad_path, 'images'))
    creator.create(data)
