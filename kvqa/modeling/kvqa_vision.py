import math
import os
from pathlib import Path

import cv2
import requests
import torch
from PIL import Image
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from scene_graph_benchmark.config import sg_cfg
from tqdm import tqdm

import kvqa.util.modeling as zm


class FeatureExtractor:
    def __init__(self, model_path, batch_size=16, device=None):
        self.device = zm.detect_cuda_device(device)
        self.batch_size = batch_size
        self.files_dir = Path(model_path)
        self.files_dir.mkdir(parents=True, exist_ok=True)

    def extract_features_save(self, img_paths, save_path):
        raise NotImplementedError


class ObjectFeatureExtractor(FeatureExtractor):
    """
    Кодирование изображений на основе модели X-152-С4,
    использовавшейся в VinVL (развитие Oscar, вывод совместим со стандартным Oscar).
    В качестве движка модели используется публичный конфиг модели
    для движка maskrcnn_benchmark (c) Facebook
    в версии scene_graph_benchmark (c) Microsoft
    https://github.com/microsoft/scene_graph_benchmark
    Основано на скрипте из MMF (c) Facebook research
    https://github.com/facebookresearch/mmf/blob/main/tools/scripts/features/extract_features_vinvl.py
    """

    def __init__(self, model_path='vinvl_vision', batch_size=16, device=None):
        super().__init__(model_path, batch_size, device)

        model_output = self.files_dir / 'model_output'

        self.model_config_file = None
        self.model_weights_file = None
        self._setup_model_files(self.files_dir)
        self.detection_model = self._build_detection_model(model_output)
        self.transforms = build_transforms(cfg, is_train=False)

    def free(self):
        if self.detection_model is not None:
            del self.detection_model

    def _setup_model_files(self, files_dir):
        pth = files_dir / 'vinvl_vg_x152c4.pth'
        if not pth.is_file():
            _download('https://dl.fbaipublicfiles.com/mmf/data/models/vinvl/detection/vinvl_vg_x152c4.pth', pth)
        self.model_weights_file = pth
        yaml = files_dir / 'vinvl_vg_x152c4.yaml'
        if not yaml.is_file():
            _download('https://dl.fbaipublicfiles.com/mmf/data/models/vinvl/detection/vinvl_x152c4.yaml', yaml)
        self.model_config_file = yaml

    def _build_detection_model(self, output_dir):
        output_dir = output_dir.as_posix() if isinstance(output_dir, Path) else str(output_dir)

        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        cfg.merge_from_file(str(self.model_config_file))
        cfg.merge_from_list([
            "MODEL.WEIGHT", str(self.model_weights_file),
            "MODEL.ROI_HEADS.NMS_FILTER", 1,
            "MODEL.ROI_HEADS.SCORE_THRESH", 0.2,
            "TEST.IGNORE_BOX_REGRESSION", False,
            "MODEL.ATTRIBUTE_ON", True,
            "TEST.OUTPUT_FEATURE", True,
            "TEST.OUTPUT_RELATION_FEATURE", True,
            "TEST.TSV_SAVE_SUBSET", ["rect", "class", "conf", "feature", "relation_feature"],
            "TEST.GATHER_ON_CPU", True,
            "MODEL.DEVICE", self.device,
            "OUTPUT_DIR", output_dir
        ])
        cfg.freeze()

        model = AttrRCNN(cfg)
        model.to(self.device)
        model.eval()
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
        checkpointer.load(cfg.MODEL.WEIGHT)
        return model

    def read_image(self, img_file):
        return self._transform_image(cv2.imread(str(img_file)))

    def read_images(self, img_files):
        loaded = list(map(lambda x: self.read_image(x), img_files))
        unzipped = tuple(zip(*loaded))
        return list(unzipped[0]), list(unzipped[1])

    def _transform_image(self, img):
        img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img, _ = self.transforms(img, target=None)
        img = img.to(self.device)
        return img, {"width": img_width, "height": img_height}

    def extract_features(self, images, infos):
        current_img_list = to_image_list(images, size_divisible=32)
        current_img_list = current_img_list.to(self.device)

        torch.manual_seed(0)
        with torch.no_grad():
            output = self.detection_model(current_img_list)

        return _to_tensors(output, infos)

    def extract_features_save(self, img_paths, save_path, save_infos=False):
        all_feats = []
        all_infos = []

        save_path = Path(save_path)
        total = math.ceil(len(img_paths) / self.batch_size)
        batches = tqdm(_batches(img_paths, self.batch_size), total=total, desc='Extracting features', unit='batch')
        for i, batch in enumerate(batches):
            images, infos = self.read_images(batch)
            feats, infos = self.extract_features(images, infos)
            for idx, file in enumerate(batch):
                feat_file = save_path / (file.name + '.pt')
                if feat_file.exists(): os.remove(feat_file)
                torch.save(feats[idx], feat_file)
                if save_infos:
                    inf_file = save_path / (file.name + '.inf.pt')
                    if inf_file.exists(): os.remove(inf_file)
                    torch.save(infos[idx], inf_file)

            all_feats += feats
            all_infos += infos
        return all_feats, all_infos


def _batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:min(i + batch_size, len(data))]


def _norm_bbox(bbox, w, h):
    bbox_aug = torch.zeros(bbox.size(0), 6)
    bbox_aug[:, :4] = bbox
    bbox_aug[:, 0] /= w
    bbox_aug[:, 1] /= h
    bbox_aug[:, 2] /= w
    bbox_aug[:, 3] /= h
    bbox_aug[:, 4] = bbox_aug[:, 2] - bbox_aug[:, 0]
    bbox_aug[:, 5] = bbox_aug[:, 3] - bbox_aug[:, 1]
    return bbox_aug


def _download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        os.remove(path)
        raise IOError(f'Something went wrong while downloading {url}')


def _to_tensors(bbox_list, infos):
    feat_list = []
    info_list = []
    for boxes, info in zip(bbox_list, infos):
        w, h = info['width'], info['height']
        boxes = boxes.to('cpu').resize((w, h))
        new_info = {k: boxes.get_field(k) for k in boxes.fields()}
        bbox = boxes.bbox
        bbox_aug = _norm_bbox(bbox, w, h)
        new_info["bbox"] = bbox_aug
        new_info["image_width"] = w
        new_info["image_height"] = h
        features = torch.cat([new_info["box_features"], new_info["bbox"]], dim=1)

        feat_list.append(features)
        info_list.append(new_info)
    return feat_list, info_list
