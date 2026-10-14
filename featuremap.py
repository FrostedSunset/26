import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, tqdm
import torchvision.transforms as T
import numpy as np
np.random.seed(0)
from engine.core import YAMLConfig
from PIL import Image

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

def get_param_by_string(model, param_str):
    keys = param_str.split('.')
    
    param = model
    for key in keys:  
        if key.isdigit():  
            key = int(key) 
            param = param[key]
        else:
            param = getattr(param, key)  

    return param

class deim_featuremap:
    handles = []
    activations = []
    def __init__(self, config, weight, layer, device, isUltralytics) -> None:
        device = torch.device(device)

        model = self.init_model(config, weight)
        model.to(device)
        model.eval()

        if isUltralytics:
            target_layers = []
            for l in layer:
                target_layers.append(eval(l))
        else:
            target_layers = [get_param_by_string(model, l) for l in layer]
        for l in target_layers:
            self.handles.append(l.register_forward_hook(self.save_activation))
        
        self.__dict__.update(locals())
    
    def init_model(self, config, weight):
        cfg = YAMLConfig(config, resume=weight)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
        checkpoint = torch.load(weight, map_location='cpu')
        if checkpoint.get('name', None) != None:
            CLASS_NAME = checkpoint['name']
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        
        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)
        return cfg.model

    def save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach())
    
    def show_layer(self):
        for name, module in self.model.named_modules():
            print(BLUE + f"Layer Name: " + ORANGE + name + BLUE + ", Layer Type: ", ORANGE, module.__class__.__name__, RESET)

    def process(self, img_path, save_path):
        file_name = os.path.basename(img_path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        save_path = os.path.join(save_path, file_name_without_extension)
        os.makedirs(save_path, exist_ok=True)

        # img process
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(self.device)

        self.model(im_data)

        for i, featmap in enumerate(self.activations):
            if len(featmap.size()) != 4:
                print(RED + f"Warning... {featmap.size()} size illegal... skip." + RESET)
                continue       
            feature_save_path = os.path.join(save_path, self.layer[i])
            os.makedirs(feature_save_path, exist_ok=True)
            for i in range(featmap.size(1)):
                feature_map = featmap[0, i]
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255
                feature_map = feature_map.detach().cpu().numpy().astype(np.uint8)
                feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_AUTUMN)
                feature_map = Image.fromarray(feature_map)
                feature_map.save(os.path.join(feature_save_path, f'{i}.png'))

        self.activations.clear()

    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in tqdm.tqdm(os.listdir(img_path)):
                if os.path.splitext(img_path_)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.process(f'{img_path}/{img_path_}', save_path)
        else:
            self.process(img_path, save_path)

def get_params():
    params = {
        'config': 'configs/test/deim_hgnetv2_s_visdrone.yml',
        'weight': '../best_stg2.pth',
        'device': 'cuda:0',
        'layer': ['backbone.stem', 'backbone.stages.0'],
        # 'layer': ['model.backbone[0]', 'model.encoder[0]'],
        'isUltralytics': False 
    }
    return params

if __name__ == '__main__':
    model = deim_featuremap(**get_params())
    model.show_layer() 
    # model(r'/home/dataset/dataset_visdrone/VisDrone2019-DET-test-challenge/images', 'featuremap_result')