import xml.etree.ElementTree as ET

# Caminho para o arquivo XML
path_to_file = r'C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\JAAD\annotations_attributes\video_0001_attributes.xml'

# Carregar o arquivo XML
tree = ET.parse(path_to_file)
root = tree.getroot()

# Caminho para salvar o arquivo de saída
output_file = r'C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\JAAD\annotations_attributes\organized_ped_attributes.txt'

with open(output_file, 'w') as file:
    # Para cada elemento 'pedestrian' no XML
    for pedestrian in root.findall(".//pedestrian"):
        # Extrai os atributos principais do pedestre
        ped_id = pedestrian.get('id')
        old_id = pedestrian.get('old_id')
        age = pedestrian.get('age')
        gender = pedestrian.get('gender')
        group_size = pedestrian.get('group_size')

        # Escreve um cabeçalho para cada pedestre
        file.write(f"Pedestrian ID: {ped_id}\n")
        file.write(f"Old ID: {old_id}\n")
        file.write(f"Age: {age}\n")
        file.write(f"Gender: {gender}\n")
        file.write(f"Group Size: {group_size}\n")
        file.write("Attributes:\n")

        # Itera sobre os atributos do pedestre e escreve-os no arquivo
        for attr, value in pedestrian.attrib.items():
            if attr not in ['id', 'old_id', 'age', 'gender', 'group_size']:  # Ignora os atributos principais
                file.write(f"  {attr}: {value}\n")

        file.write("="*40 + "\n\n")  # Linha divisória entre pedestres
