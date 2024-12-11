from utils import *
import os

# Defina diretamente os caminhos aqui
data = "JAAD"
gt_file = r"C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\JAAD\images\ground_truth_teste.txt"
txt_out = r"C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\output_txt_with_ann"

# Execução do código com os caminhos definidos diretamente
if data == 'JAAD':
    jaad_splitter = JAAD_splitter(gt_file, txt_out)
    #jaad_splitter.split_(jaad_splitter.data, 'scenes')
    jaad_splitter.split_with_ann_(jaad_splitter.data, 'scenes')
    #jaad_splitter.split_(jaad_splitter.data, 'instances')
    jaad_splitter.split_with_ann_(jaad_splitter.data, 'instances')
else:
    pie_splitter = PIE_splitter(gt_file, txt_out)
    pie_splitter.split_(pie_splitter.data, 'scenes')
    pie_splitter.split_(pie_splitter.data, 'instances')
