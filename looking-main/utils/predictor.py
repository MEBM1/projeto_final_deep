import configparser
import os, errno
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import argparse
import PIL
from zipfile import ZipFile
from glob import glob
from tqdm import tqdm
import openpifpaf.datasets as datasets
import time

#from utils.dataset import *
#from utils.network import *
#from utils.utils_predict import *

from dataset import *
from network import *
from utils_predict import *

from PIL import Image, ImageFile

DOWNLOAD = None
INPUT_SIZE=51
FONT = cv2.FONT_HERSHEY_SIMPLEX

ImageFile.LOAD_TRUNCATED_IMAGES = True

print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)


class Predictor():
    """
        Class definition for the predictor.
    """
    def __init__(self, args, pifpaf_ver='shufflenetv2k30'):
        device = args.device
        args.checkpoint = pifpaf_ver
        args.force_complete_pose = True
        if device != 'cpu':
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(device) if use_cuda else "cpu")
        else:
            self.device = torch.device('cpu')
        args.device = self.device
        print('device : {}'.format(self.device))
        self.path_images = args.images
        #self.net, self.processor, self.preprocess = load_pifpaf(args)
        self.predictor_ = load_pifpaf(args)
        self.path_model = './models/predictor'
        try:
            os.makedirs(self.path_model)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.mode = args.mode
        self.model = self.get_model().to(self.device)
        if args.image_output is None:
            self.path_out = './output'
            self.path_out = filecreation(self.path_out)
        else:
            self.path_out = args.image_output
        self.track_time = args.time
        if self.track_time:
            self.pifpaf_time = []
            self.inference_time = []
            self.total_time = []

    
    def get_model(self):
        if self.mode == 'joints':
            model = LookingModel(INPUT_SIZE)
            print(self.device)
            if not os.path.isfile(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p')):
                """
                DOWNLOAD(LOOKING_MODEL, os.path.join(self.path_model, 'Looking_Model.zip'), quiet=False)
                with ZipFile(os.path.join(self.path_model, 'Looking_Model.zip'), 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall()
                exit(0)"""
                raise NotImplementedError
            model.load_state_dict(torch.load(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p'), map_location=self.device))
            model.eval()
        else:
            model = AlexNet_head(self.device)
            if not os.path.isfile(os.path.join(self.path_model, 'AlexNet_LOOK.p')):
                """
                DOWNLOAD(LOOKING_MODEL, os.path.join(self.path_model, 'Looking_Model.zip'), quiet=False)
                with ZipFile(os.path.join(self.path_model, 'Looking_Model.zip'), 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall()
                exit(0)"""
                raise NotImplementedError
            model.load_state_dict(torch.load(os.path.join(self.path_model, 'AlexNet_LOOK.p')))
            model.eval()
        return model

    def predict_look(self, boxes, keypoints, im_size, batch_wise=True):
        label_look = []
        final_keypoints = []
        if batch_wise:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
                    kps_final = np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    #X, Y = normalize(X, Y, divide=True, height_=False)
                    kps_final_normalized = np.array([X, Y, kps_final[34:]]).flatten().tolist()
                    final_keypoints.append(kps_final_normalized)
                tensor_kps = torch.Tensor([final_keypoints]).to(self.device)
                if self.track_time:
                    start = time.time()
                    out_labels = self.model(tensor_kps.squeeze(0)).detach().cpu().numpy().reshape(-1)
                    end = time.time()
                    self.inference_time.append(end-start)
                else:
                    out_labels = self.model(tensor_kps.squeeze(0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        else:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
                    kps_final = np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    #X, Y = normalize(X, Y, divide=True, height_=False)
                    kps_final_normalized = np.array([X, Y, kps_final[34:]]).flatten().tolist()
                    #final_keypoints.append(kps_final_normalized)
                    tensor_kps = torch.Tensor(kps_final_normalized).to(self.device)
                    if self.track_time:
                        start = time.time()
                        out_labels = self.model(tensor_kps.unsqueeze(0)).detach().cpu().numpy().reshape(-1)
                        end = time.time()
                        self.inference_time.append(end-start)
                    else:
                        out_labels = self.model(tensor_kps.unsqueeze(0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        return out_labels
    
    def predict_look_alexnet(self, boxes, image, batch_wise=True):
        out_labels = []
        data_transform = transforms.Compose([
                        SquarePad(),
                        transforms.Resize((227,227)),
                    transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
        if len(boxes) != 0:
            if batch_wise:
                heads = []
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    x1, y1, x2, y2, _ = bbox
                    w, h = abs(x2-x1), abs(y2-y1)
                    head_image = Image.fromarray(np.array(image)[int(y1):int(y1+(h/3)), int(x1):int(x2), :])
                    head_tensor = data_transform(head_image)
                    heads.append(head_tensor.detach().cpu().numpy())
                if self.track_time:
                    start = time.time()
                    out_labels = self.model(torch.Tensor([heads]).squeeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)
                    end = time.time()
                    self.inference_time.append(end-start)
            else:
                out_labels = []
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    x1, y1, x2, y2, _ = bbox
                    w, h = abs(x2-x1), abs(y2-y1)
                    head_image = Image.fromarray(np.array(image)[int(y1):int(y1+(h/3)), int(x1):int(x2), :])
                    head_tensor = data_transform(head_image)
                    #heads.append(head_tensor.detach().cpu().numpy())
                    if self.track_time:
                        start = time.time()
                        looking_label = self.model(torch.Tensor(head_tensor).unsqueeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)[0]
                        end = time.time()
                        self.inference_time.append(end-start)
                    else:
                        looking_label = self.model(torch.Tensor(head_tensor).unsqueeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)[0]
                    out_labels.append(looking_label)
                #if self.track_time:
                #    out_labels = self.model(torch.Tensor([heads]).squeeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)
        else:
            out_labels = []
        return out_labels
    
    def render_image_keypoints_with_bbox(self, image, bbox, keypoints, pred_labels, image_name, transparency, eyecontact_thresh, bb_gt):
        # Convertendo a imagem PIL para OpenCV
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convertendo de RGB para BGR

        # Escala para o tamanho da fonte (se necessário)
        scale = 0.007
        imageWidth, imageHeight, _ = open_cv_image.shape
        font_scale = min(imageWidth, imageHeight) / (10 / scale)

        # Criando uma máscara para desenhar os keypoints
        mask = np.zeros(open_cv_image.shape, dtype=np.uint8)

        # Desenhando os keypoints sem considerar o eyecontact_thresh
        for i, _ in enumerate(pred_labels):  # Ignoramos pred_labels para não verificar o olhar
            if i < len(keypoints):  # Garantir que o índice existe
                if keypoints[i]:  # Verificar se keypoints[i] não está vazio ou inválido
                    print(f"Keypoints {i}: {keypoints[i]}")  # Depuração para garantir que os keypoints estão corretos
                    # Assumimos uma cor padrão (por exemplo, verde) para os keypoints
                    color = (0, 255, 0)  # Verde
                    mask = draw_skeleton(mask, keypoints[i], color)  # Desenhando os keypoints na máscara
            
        for box in bbox:  # Itera sobre cada bounding box
            print(box)
            print('Desenhando bounding box...')
            
            if len(box) >= 4:  # Garante que há ao menos 4 valores para o bounding box
                x_min, y_min, x_max, y_max = map(int, box[:4])  # Considera apenas os 4 primeiros valores
                color_bbox = (0, 0, 255)  # Vermelho para o bounding box
                thickness = 2  # Espessura da linha do bounding box
                cv2.rectangle(open_cv_image, (x_min, y_min), (x_max, y_max), color_bbox, thickness)

       # for bbox_gt in bb_gt:  # Itera sobre cada bounding box
        print('bbox gt renderizando', bb_gt)  
        if len(bb_gt) >= 4:  # Garante que há ao menos 4 valores para o bounding box
            x_min, y_min, x_max, y_max = map(int, bb_gt[:4])  # Considera apenas os 4 primeiros valores
            color_bbox = (255, 0, 0)  # Vermelho para o bounding box
            thickness = 2  # Espessura da linha do bounding box
            cv2.rectangle(open_cv_image, (x_min, y_min), (x_max, y_max), color_bbox, thickness)

        # Aplica erodimento e desfocagem na máscara
        mask = cv2.erode(mask, (7, 7), iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Adiciona a máscara à imagem original com a transparência especificada
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, transparency, 1.0)

        # Salva a imagem final com os keypoints e bounding box desenhados
        cv2.imwrite(os.path.join(self.path_out, image_name[:-4] + '.predictions.png'), open_cv_image)

        
    def render_image_keypoints(self, image, bbox, keypoints, pred_labels, image_name, transparency, eyecontact_thresh):
        # Convertendo a imagem PIL para OpenCV
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convertendo de RGB para BGR

        # Escala para o tamanho da fonte (se necessário)
        scale = 0.007
        imageWidth, imageHeight, _ = open_cv_image.shape
        font_scale = min(imageWidth, imageHeight) / (10 / scale)

        # Criando uma máscara para desenhar os keypoints
        mask = np.zeros(open_cv_image.shape, dtype=np.uint8)

        # Desenhando os keypoints sem considerar o eyecontact_thresh
        for i, _ in enumerate(pred_labels):  # Ignoramos pred_labels para não verificar o olhar
            if i < len(keypoints):  # Garantir que o índice existe
                if keypoints[i]:  # Verificar se keypoints[i] não está vazio ou inválido
                    print(f"Keypoints {i}: {keypoints[i]}")  # Depuração para garantir que os keypoints estão corretos
                    # Assumimos uma cor padrão (por exemplo, verde) para os keypoints
                    color = (0, 255, 0)  # Verde
                    mask = draw_skeleton(mask, keypoints[i], color)  # Desenhando os keypoints na máscara

        # Aplica erodimento e desfocagem na máscara
        mask = cv2.erode(mask, (7, 7), iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Adiciona a máscara à imagem original com a transparência especificada
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, transparency, 1.0)

        # Salva a imagem final com os keypoints desenhados
        cv2.imwrite(os.path.join(self.path_out, image_name[:-4] + '.predictions.png'), open_cv_image)

    def render_image(self, image, bbox, keypoints, pred_labels, image_name, transparency, eyecontact_thresh):
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        scale = 0.007
        imageWidth, imageHeight, _ = open_cv_image.shape
        font_scale = min(imageWidth,imageHeight)/(10/scale)


        mask = np.zeros(open_cv_image.shape, dtype=np.uint8)
        for i, label in enumerate(pred_labels):

            if label > eyecontact_thresh:
                color = (0,255,0)
            else:
                color = (0,0,255)
            mask = draw_skeleton(mask, keypoints[i], color)
        mask = cv2.erode(mask,(7,7),iterations = 1)
        mask = cv2.GaussianBlur(mask,(3,3),0)
        #open_cv_image = cv2.addWeighted(open_cv_image, 0.5, np.ones(open_cv_image.shape, dtype=np.uint8)*255, 0.5, 1.0)
        #open_cv_image = cv2.addWeighted(open_cv_image, 0.5, np.zeros(open_cv_image.shape, dtype=np.uint8), 0.5, 1.0)
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, transparency, 1.0)
        cv2.imwrite(os.path.join(self.path_out, image_name[:-4]+'.predictions.png'), open_cv_image)


    def predict(self, args):
        transparency = args.transparency
        eyecontact_thresh = args.looking_threshold
        
        if args.glob:
            array_im = glob(os.path.join(args.images[0], '*'+args.glob))
        else:
            array_im = args.images
        loader = self.predictor_.images(array_im)
        start_pifpaf = time.time()
        for i, (pred_batch, _, meta_batch) in enumerate(tqdm(loader)):
            if self.track_time:
                end_pifpaf = time.time()
                self.pifpaf_time.append(end_pifpaf-start_pifpaf)
            cpu_image = PIL.Image.open(open(meta_batch['file_name'], 'rb')).convert('RGB')
            pifpaf_outs = {
            'json_data': [ann.json_data() for ann in pred_batch],
            'image': cpu_image
            }
            
            im_name = os.path.basename(meta_batch['file_name'])
            im_size = (cpu_image.size[0], cpu_image.size[1])
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['json_data'], im_size, enlarge_boxes=False)
            if self.mode == 'joints':
                pred_labels = self.predict_look(boxes, keypoints, im_size)
            else:
                pred_labels = self.predict_look_alexnet(boxes, cpu_image)
            if self.track_time:
                end_process = time.time()
                self.total_time.append(end_process - start_pifpaf)
            
            #break
            if self.track_time:
                start_pifpaf = time.time()
            else:
                self.render_image(pifpaf_outs['image'], boxes, keypoints, pred_labels, im_name, transparency, eyecontact_thresh)

            #if i > 20:
            #    break
        
        if self.track_time and len(self.pifpaf_time) != 0 and len(self.inference_time) != 0:
            print('Av. pifpaf time : {} ms. ± {} ms'.format(np.mean(self.pifpaf_time)*1000, np.std(self.pifpaf_time)*1000))
            print('Av. inference time : {} ms. ± {} ms'.format(np.mean(self.inference_time)*1000, np.std(self.inference_time)*1000))
            print('Av. total time : {} ms. ± {} ms'.format(np.mean(self.total_time)*1000, np.std(self.total_time)*1000))
    
### mais uma funcao teste
    def teste_pifpaf(self):
        print('teste pifpaf')

        #image_path = 'C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\JAAD\\images\\video_0209\\00075.png'
        image_path = 'C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\JAAD\\images\\video_0135\\00080_augmented.png'
        #image_path = 'C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\JAAD\\images\\video_0190\\00010.png'
        #image_path = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\JAAD\\images\\video_0340\\00140_augmented.png"
       # image_path = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\looking-main\\images\\00477_augmented.png"
        cpu_image = PIL.Image.open(image_path).convert('RGB')
        im_name = os.path.basename(image_path)
        im_size = (cpu_image.size[0], cpu_image.size[1])
            
        # Chamar o OpenPifPaf para predições
        loader = self.predictor_.images([image_path])
        print('loader', loader)

        for pred_batch, _, meta_batch in loader:
            print('pred_batch',pred_batch)
            print('meta batch', meta_batch)
            pifpaf_outs = {
                    'json_data': [ann.json_data() for ann in pred_batch],
                    'image': cpu_image
                }
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['json_data'], im_size, enlarge_boxes=False)
            pred_labels = self.predict_look(boxes, keypoints, im_size)
            predictions = []
            print('keypoints', len(keypoints))
            for i in range(len(keypoints)):
                print('oi')
                keypoint = keypoints[i]
                bbox = boxes[i]
                score = pred_labels[i]  # Presume-se que isso já seja a pontuação de confiança ou rótulo predito

                # Reformata os keypoints em uma única lista [x1, y1, v1, x2, y2, v2, ...]
                flattened_keypoints = [coord for kp in keypoint for coord in kp]

                # Criando um dicionário para cada predição no formato desejado
                prediction = {
                    'keypoints': flattened_keypoints,  # Keypoints achatados
                    'bbox': [float(coord) for coord in bbox],  # Garante que bbox seja uma lista de floats
                    'score': float(score),  # Garante que score seja float
                    'category_id': 1  # Valor fixo para a categoria (se aplicável)
                }

                # Adiciona ao array de predições
                predictions.append(prediction)

            print('predictions', predictions)
               #print('predictionnnn', prediction)
                # Gerar o JSON com as predições
            json_predictions = json.dumps({"predictions": predictions}, indent=4)
            #json_predictions = predictions#json.dumps(predictions, indent=4)   
                # Aqui você pode retornar ou salvar o JSON gerado
            output_json_path = os.path.join(self.path_out, im_name[:-4] + '.predictions.json')
            with open(output_json_path, 'w') as json_file:
                json_file.write(json_predictions)

            print(f"JSON de predições salvo em {output_json_path}")
            #return json_predictions
            return predictions
#funçao teste
    def process_augmented_image(self, image_path, transparency, eyecontact_thresh, bb_gt):
        """
        Processa uma única imagem que contém 'augmented' no nome e gera os keypoints e outros dados.
        
        Args:
            image_path (str): Caminho da imagem a ser processada.
            transparency (float): Transparência para renderização.
            eyecontact_thresh (float): Threshold para contato visual.
        """
        if 'augmented' not in image_path:
            return  # Ignora se o nome não contém 'augmented'

        try:
            # Carregar imagem
            cpu_image = PIL.Image.open(image_path).convert('RGB')
            im_name = os.path.basename(image_path)
            im_size = (cpu_image.size[0], cpu_image.size[1])
            
            # Chamar o OpenPifPaf para predições
            loader = self.predictor_.images([image_path])
            for pred_batch, _, meta_batch in loader:
                if not pred_batch:  # Verifica se pred_batch está vazio
                    print(f"Não foi possível fazer a detecção de pose para {image_path}. Pulando para a próxima.")
                    continue  # Ignora este lote e segue para o próximo, se houver
                
                print('pred_batch', pred_batch)
                print('meta batch', meta_batch)
                
                pifpaf_outs = {
                    'json_data': [ann.json_data() for ann in pred_batch],
                    'image': cpu_image
                }
                boxes, keypoints = preprocess_pifpaf(pifpaf_outs['json_data'], im_size, enlarge_boxes=False)
                pred_labels = self.predict_look(boxes, keypoints, im_size)
                predictions = []
                print('keypoints', len(keypoints))
                for i in range(len(keypoints)):
                    keypoint = keypoints[i]
                    bbox = boxes[i]
                    score = pred_labels[i]  # Presume-se que isso já seja a pontuação de confiança ou rótulo predito

                    # Reformata os keypoints em uma única lista [x1, y1, v1, x2, y2, v2, ...]
                    flattened_keypoints = [coord for kp in keypoint for coord in kp]

                    # Criando um dicionário para cada predição no formato desejado
                    prediction = {
                        'keypoints': flattened_keypoints,  # Keypoints achatados
                        'bbox': [float(coord) for coord in bbox],  # Garante que bbox seja uma lista de floats
                        'score': float(score),  # Garante que score seja float
                        'category_id': 1  # Valor fixo para a categoria (se aplicável)
                    }

                    # Adiciona ao array de predições
                    predictions.append(prediction)

                # Gerar o JSON com as predições
                #json_predictions = json.dumps({"predictions": predictions}, indent=4)
                json_predictions = json.dumps(predictions, indent=4)   
                
                # Aqui você pode retornar ou salvar o JSON gerado
                #output_json_path = os.path.join(self.path_out, im_name[:-4] + '.predictions.json')
                print('img name', im_name)
                print('img path',image_path)
                
                path_joints = "C:\\Users\\madua\\Documents\\Mestrado\\Deep Learning\\Projeto Final\\output_pifpaf_01_2k30\\output_pifpaf_01_2k30"
                filename = os.path.splitext(image_path)[0]  # Remove extensão
                result = "\\".join(filename.split(os.sep)[-2:])  # Últimas duas partes do caminho

                output_json_path = os.path.join(path_joints, result + '.predictions.json')
                with open(output_json_path, 'w') as json_file:
                    json_file.write(json_predictions)

                #print(f"JSON de predições salvo em {output_json_path}")
                #print('renderizeiii', pifpaf_outs['image'] )
                #self.render_image_keypoints_with_bbox(pifpaf_outs['image'], boxes, keypoints, pred_labels, im_name, transparency, eyecontact_thresh, bb_gt)
                #return json_predictions
                return predictions

        except Exception as e:
            print(f"Erro ao processar a imagem {image_path}: {e}")
            return None



    def generate_img_w_keypoints(self, args):
        transparency = args.transparency
        eyecontact_thresh = args.looking_threshold
        
        if args.glob:
            array_im = glob(os.path.join(args.images[0], '*'+args.glob))
        else:
            array_im = args.images
        loader = self.predictor_.images(array_im)
        start_pifpaf = time.time()
        for i, (pred_batch, _, meta_batch) in enumerate(tqdm(loader)):
            if self.track_time:
                end_pifpaf = time.time()
                self.pifpaf_time.append(end_pifpaf-start_pifpaf)
            cpu_image = PIL.Image.open(open(meta_batch['file_name'], 'rb')).convert('RGB')
            pifpaf_outs = {
            'json_data': [ann.json_data() for ann in pred_batch],
            'image': cpu_image
            }
            
            im_name = os.path.basename(meta_batch['file_name'])
            im_size = (cpu_image.size[0], cpu_image.size[1])
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['json_data'], im_size, enlarge_boxes=False)
            if self.mode == 'joints':
                pred_labels = self.predict_look(boxes, keypoints, im_size)
            else:
                pred_labels = self.predict_look_alexnet(boxes, cpu_image)
            if self.track_time:
                end_process = time.time()
                self.total_time.append(end_process - start_pifpaf)
            
            #break
            if self.track_time:
                start_pifpaf = time.time()
            else:
                self.render_image_keypoints(pifpaf_outs['image'], boxes, keypoints, pred_labels, im_name, transparency, eyecontact_thresh)

            #if i > 20:
            #    break
        
        if self.track_time and len(self.pifpaf_time) != 0 and len(self.inference_time) != 0:
            print('Av. pifpaf time : {} ms. ± {} ms'.format(np.mean(self.pifpaf_time)*1000, np.std(self.pifpaf_time)*1000))
            print('Av. inference time : {} ms. ± {} ms'.format(np.mean(self.inference_time)*1000, np.std(self.inference_time)*1000))
            print('Av. total time : {} ms. ± {} ms'.format(np.mean(self.total_time)*1000, np.std(self.total_time)*1000))
    