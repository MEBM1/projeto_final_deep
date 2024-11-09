import xml.etree.ElementTree as ET

# Caminho para o arquivo XML
path_to_file = r'C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\JAAD\annotations_appearance\video_0001_appearance.xml'

# Carregar o arquivo XML
tree = ET.parse(path_to_file)
root = tree.getroot()

# Caminho para salvar o arquivo de saída
output_file = r'C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\JAAD\annotations_appearance\organized_output.txt'

with open(output_file, 'w') as file:
    for track in root.findall(".//track"):  # Encontra cada "track" no XML
        track_id = track.get('id')
        label = track.get('label')
        old_id = track.get('old_id')

        # Escreve o cabeçalho da track
        file.write(f"Track ID: {track_id}\n")
        file.write(f"Label: {label}\n")
        file.write(f"Old ID: {old_id}\n")
        file.write("="*40 + "\n")  # Linha divisória

        for box in track.findall(".//box"):  # Encontra cada "box" dentro da "track"
            frame = box.get('frame')
            file.write(f"  Frame: {frame}\n")
            file.write("  Attributes:\n")

            # Itera sobre os atributos do box e os escreve com indentação
            for attr, value in box.attrib.items():
                if attr != "frame":  # Pula o frame, pois já o escrevemos
                    file.write(f"    {attr}: {value}\n")
            
            file.write("-"*30 + "\n")  # Linha divisória para cada box

        file.write("\n")  # Adiciona uma linha em branco entre tracks
