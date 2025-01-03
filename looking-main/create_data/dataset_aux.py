from utils import JAAD_loader, JAAD_creator, Kitti_creator, PIE_loader, PIE_creator
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import sys
sys.path.insert(0,r'C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\looking-main\utils')
from predictor import Predictor
#### teste

from predictor import *

parser = argparse.ArgumentParser(prog='python3 predict', usage='%(prog)s [options] images', description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--version', action='version',version='Looking Model {version}'.format(version=0.1))
parser.add_argument('--images', nargs='*',help='input images')
parser.add_argument('--transparency', default=0.4, type=float, help='transparency of the overlayed poses')
parser.add_argument('--looking_threshold', default=0.5, type=float, help='eye contact threshold')
parser.add_argument('--mode', default='joints', type=str, help='prediction mode')
parser.add_argument('--time', action='store_true', help='track comptutational time')
parser.add_argument('--glob', help='glob expression for input images (for many images)')

# Pifpaf args

parser.add_argument('-o', '--image-output', default=None, nargs='?', const=True, help='Whether to output an image, with the option to specify the output path or directory')
parser.add_argument('--json-output', default=None, nargs='?', const=True,help='Whether to output a json file, with the option to specify the output path or directory')
parser.add_argument('--batch_size', default=1, type=int, help='processing batch size')
parser.add_argument('--device', default='0', type=str, help='cuda device')
parser.add_argument('--long-edge', default=None, type=int, help='rescale the long side of the image (aspect ratio maintained)')
parser.add_argument('--loader-workers', default=None, type=int, help='number of workers for data loading')
parser.add_argument('--precise-rescaling', dest='fast_rescaling', default=True, action='store_false', help='use more exact image rescaling (requires scipy)')
parser.add_argument('--checkpoint_', default='shufflenetv2k30', type=str, help='backbone model to use')
parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')

decoder.cli(parser)
logger.cli(parser)
network.Factory.cli(parser)
show.cli(parser)
visualizer.cli(parser)

args = parser.parse_args()


predictor = Predictor(args)

######
# Definindo os caminhos diretamente
jaad_path = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\JAAD"
path_joints = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\output_pifpaf_01_2k30\\output_pifpaf_01_2k30"
txt_out = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\output_txt_test"
dir_out = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\pasta_aux - Copia"
data = "Augmentation_JAAD"

# Carregando e criando o dataset JAAD


if 'Augmentation_JAAD' in data:

    jaad_loader = JAAD_loader(jaad_path, txt_out)
    data_dict = jaad_loader.generate_with_attributes_ann(r"C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\JAAD\images")


# Verifique o tipo de data_dict para garantir que é um dicionário
    #print(f"Tipo de data_dict: {type(data_dict)}")  # Deve ser <class 'dict'>
    #print(f"Chaves de data_dict: {data_dict.keys()}")  # Deve ter as chaves ['path', 'bbox', 'names', 'Y', 'video', 'attributes', 'image']
    #print(int(len(data_dict['path']) * 0.3))
    #print('data dict path',data_dict['path'])
    data_aux = data_dict
    augmented_data = jaad_loader.augment_selected_samples(data_aux)
    #print('augmented dataaaa', augmented_data)
    jaad_creator = JAAD_creator(txt_out, dir_out, path_joints, r"C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\JAAD\images")
    jaad_creator.create_with_ann_augmented(augmented_data, predictor)
    
elif 'JAAD' in data:
    jaad_loader = JAAD_loader(jaad_path, txt_out)
    #dict_annotation = jaad_loader.generate()
    dict_annotation = jaad_loader.generate_with_attributes_ann()
    jaad_creator = JAAD_creator(txt_out, dir_out, path_joints, os.path.join(jaad_path, 'images'))
    #jaad_creator.create(dict_annotation)
    jaad_creator.create_with_ann(dict_annotation)


# Carregando e criando o dataset PIE (caso necessário)
elif 'PIE' in data:
    pie_loader = PIE_loader(jaad_path)
    data = pie_loader.generate()

    creator = PIE_creator(txt_out, dir_out, path_joints, os.path.join(jaad_path, 'images'))
    creator.create(data)
