import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from collections import OrderedDict

from feat.emo_detectors.ResMaskNet.resmasknet_test import (
    ResMasking,
)
from feat.identity_detectors.facenet.facenet_model import InceptionResnetV1
from feat.facepose_detectors.img2pose.deps.models import (
    FasterDoFRCNN,
    postprocess_img2pose,
)
from feat.au_detectors.StatLearning.SL_test import XGBClassifier, SVMClassifier
from feat.emo_detectors.StatLearning.EmoSL_test import EmoSVMClassifier
from feat.landmark_detectors.mobilefacenet_test import MobileFaceNet
from feat.landmark_detectors.basenet_test import MobileNet_GDConv
from feat.landmark_detectors.pfld_compressed_test import PFLDInference
from feat.pretrained import load_model_weights, AU_LANDMARK_MAP
from feat.utils import (
    set_torch_device,
    openface_2d_landmark_columns,
    FEAT_EMOTION_COLUMNS,
    FEAT_FACEBOX_COLUMNS,
    FEAT_FACEPOSE_COLUMNS_6D,
    FEAT_IDENTITY_COLUMNS,
)
from feat.utils.io import get_resource_path
from feat.utils.image_operations import (
    convert_image_to_tensor,
    extract_face_from_bbox_torch,
    inverse_transform_landmarks_torch,
    extract_hog_features,
    convert_bbox_output,
    compute_original_image_size,
)
from feat.data import Fex, ImageDataset, TensorDataset, VideoDataset
from feat.streaming import StreamingVideoSource, StreamingResult, FaceResult
from skops.io import load, get_untrusted_types
from safetensors.torch import load_file
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import Compose, Normalize
import sys
import warnings
from pathlib import Path

sys.modules["__main__"].__dict__["XGBClassifier"] = XGBClassifier
sys.modules["__main__"].__dict__["SVMClassifier"] = SVMClassifier
sys.modules["__main__"].__dict__["EmoSVMClassifier"] = EmoSVMClassifier

# Supress sklearn warning about pickled estimators and diff sklearn versions
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


class Detector(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        identity_model="facenet",
        device="cpu",
        optimize=False,
        half_precision=False,
    ):
        super(Detector, self).__init__()

        self.info = dict(
            face_model="img2pose",
            landmark_model=None,
            emotion_model=None,
            facepose_model="img2pose",
            au_model=None,
            identity_model=None,
        )
        self.device = set_torch_device(device)
        self._optimize = optimize
        self._half_precision = half_precision and str(self.device).startswith("cuda")

        # Load Model Configurations
        facepose_config_file = hf_hub_download(
            repo_id="py-feat/img2pose",
            filename="config.json",
            cache_dir=get_resource_path(),
        )
        with open(facepose_config_file, "r") as f:
            facepose_config = json.load(f)

        # Initialize img2pose
        backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=None)
        backbone.eval()
        backbone.to(self.device)
        self.facepose_detector = FasterDoFRCNN(
            backbone=backbone,
            num_classes=2,
            min_size=facepose_config["min_size"],
            max_size=facepose_config["max_size"],
            pose_mean=torch.tensor(facepose_config["pose_mean"]),
            pose_stddev=torch.tensor(facepose_config["pose_stddev"]),
            threed_68_points=torch.tensor(facepose_config["threed_points"]),
            rpn_pre_nms_top_n_test=facepose_config["rpn_pre_nms_top_n_test"],
            rpn_post_nms_top_n_test=facepose_config["rpn_post_nms_top_n_test"],
            bbox_x_factor=facepose_config["bbox_x_factor"],
            bbox_y_factor=facepose_config["bbox_y_factor"],
            expand_forehead=facepose_config["expand_forehead"],
        )
        facepose_model_file = hf_hub_download(
            repo_id="py-feat/img2pose",
            filename="model.safetensors",
            cache_dir=get_resource_path(),
        )
        facepose_checkpoint = load_file(facepose_model_file)
        self.facepose_detector.load_state_dict(facepose_checkpoint, load_model_weights)
        self.facepose_detector.eval()
        self.facepose_detector.to(self.device)


        # Initialize Landmark Detector
        self.info["landmark_model"] = landmark_model
        if landmark_model is not None:
            if landmark_model == "mobilefacenet":
                self.face_size = 112
                self.landmark_detector = MobileFaceNet(
                    [self.face_size, self.face_size], 136, device=self.device
                )
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/mobilefacenet",
                    filename="mobilefacenet_model_best.pth.tar",
                    cache_dir=get_resource_path(),
                )
                landmark_state_dict = torch.load(
                    landmark_model_file, map_location=self.device, weights_only=True
                )["state_dict"]  # Ensure Model weights are Float32 for MPS
            elif landmark_model == "mobilenet":
                self.face_size = 224
                self.landmark_detector = MobileNet_GDConv(136)
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/mobilenet",
                    filename="mobilenet_224_model_best_gdconv_external.pth.tar",
                    cache_dir=get_resource_path(),
                )
                mobilenet_state_dict = torch.load(
                    landmark_model_file, map_location=self.device, weights_only=True
                )["state_dict"]  # Ensure Model weights are Float32 for MPS
                landmark_state_dict = OrderedDict()
                for k, v in mobilenet_state_dict.items():
                    if "module." in k:
                        k = k.replace("module.", "")
                    landmark_state_dict[k] = v
            elif landmark_model == "pfld":
                self.face_size = 112
                self.landmark_detector = PFLDInference()
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/pfld",
                    filename="pfld_model_best.pth.tar",
                    cache_dir=get_resource_path(),
                )
                landmark_state_dict = torch.load(
                    landmark_model_file, map_location=self.device, weights_only=True
                )["state_dict"]  # Ensure Model weights are Float32 for MPS
            else:
                raise ValueError("{landmark_model} is not currently supported.")
            self.landmark_detector.load_state_dict(landmark_state_dict)
            self.landmark_detector.eval()
            self.landmark_detector.to(self.device)

        else:
            self.landmark_detector = None

        # Initialize AU Detector
        self.info["au_model"] = au_model
        if au_model is not None:
            if self.landmark_detector is not None:
                if au_model == "xgb":
                    self.au_detector = XGBClassifier()
                    au_model_path = hf_hub_download(
                        repo_id="py-feat/xgb_au",
                        filename="xgb_au_classifier.skops",
                        cache_dir=get_resource_path(),
                    )

                elif au_model == "svm":
                    self.au_detector = SVMClassifier()
                    au_model_path = hf_hub_download(
                        repo_id="py-feat/svm_au",
                        filename="svm_au_classifier.skops",
                        cache_dir=get_resource_path(),
                    )
                else:
                    raise ValueError("{au_model} is not currently supported.")

                au_unknown_types = get_untrusted_types(file=au_model_path)
                loaded_au_model = load(au_model_path, trusted=au_unknown_types)
                self.au_detector.load_weights(
                    scaler_upper=loaded_au_model.scaler_upper,
                    pca_model_upper=loaded_au_model.pca_model_upper,
                    scaler_lower=loaded_au_model.scaler_lower,
                    pca_model_lower=loaded_au_model.pca_model_lower,
                    scaler_full=loaded_au_model.scaler_full,
                    pca_model_full=loaded_au_model.pca_model_full,
                    classifiers=loaded_au_model.classifiers,
                )
            else:
                raise ValueError(
                    "Landmark Detector is required for AU Detection with {au_model}."
                )
        else:
            self.au_detector = None

        # Initialize Emotion Detector
        self.info["emotion_model"] = emotion_model
        if emotion_model is not None:
            if emotion_model == "resmasknet":
                emotion_config_file = hf_hub_download(
                    repo_id="py-feat/resmasknet",
                    filename="config.json",
                    cache_dir=get_resource_path(),
                )
                with open(emotion_config_file, "r") as f:
                    emotion_config = json.load(f)

                self.emotion_detector = ResMasking(
                    "", in_channels=emotion_config["in_channels"]
                )
                self.emotion_detector.fc = nn.Sequential(
                    nn.Dropout(0.4), nn.Linear(512, emotion_config["num_classes"])
                )
                emotion_model_file = hf_hub_download(
                    repo_id="py-feat/resmasknet",
                    filename="ResMaskNet_Z_resmasking_dropout1_rot30.pth",
                    cache_dir=get_resource_path(),
                )
                emotion_checkpoint = torch.load(
                    emotion_model_file, map_location=device, weights_only=True
                )["net"]
                self.emotion_detector.load_state_dict(emotion_checkpoint)
                self.emotion_detector.eval()
                self.emotion_detector.to(self.device)

            elif emotion_model == "svm":
                if self.landmark_detector is not None:
                    self.emotion_detector = EmoSVMClassifier()
                    emotion_model_path = hf_hub_download(
                        repo_id="py-feat/svm_emo",
                        filename="svm_emo_classifier.skops",
                        cache_dir=get_resource_path(),
                    )
                    emotion_unknown_types = get_untrusted_types(file=emotion_model_path)
                    loaded_emotion_model = load(
                        emotion_model_path, trusted=emotion_unknown_types
                    )
                    self.emotion_detector.load_weights(
                        scaler_full=loaded_emotion_model.scaler_full,
                        pca_model_full=loaded_emotion_model.pca_model_full,
                        classifiers=loaded_emotion_model.classifiers,
                    )
                else:
                    raise ValueError(
                        "Landmark Detector is required for Emotion Detection with {emotion_model}."
                    )

            else:
                raise ValueError("{emotion_model} is not currently supported.")
        else:
            self.emotion_detector = None

        # Initialize Identity Detector -  facenet
        self.info["identity_model"] = identity_model
        if identity_model is not None:
            if identity_model == "facenet":
                self.identity_detector = InceptionResnetV1(
                    pretrained=None,
                    classify=False,
                    num_classes=None,
                    dropout_prob=0.6,
                    device=self.device,
                )
                self.identity_detector.logits = nn.Linear(512, 8631)
                identity_model_file = hf_hub_download(
                    repo_id="py-feat/facenet",
                    filename="facenet_20180402_114759_vggface2.pth",
                    cache_dir=get_resource_path(),
                )
                self.identity_detector.load_state_dict(
                    torch.load(
                        identity_model_file, map_location=device, weights_only=True
                    )
                )
                self.identity_detector.eval()
                self.identity_detector.to(self.device)

            else:
                raise ValueError("{identity_model} is not currently supported.")
        else:
            self.identity_detector = None

        # Apply torch.compile optimizations if requested
        if self._optimize and hasattr(torch, "compile"):
            compile_kwargs = {"mode": "reduce-overhead"}
            if self.landmark_detector is not None:
                self.landmark_detector = torch.compile(
                    self.landmark_detector, **compile_kwargs
                )
            if self.emotion_detector is not None and emotion_model == "resmasknet":
                self.emotion_detector = torch.compile(
                    self.emotion_detector, **compile_kwargs
                )
            if self.identity_detector is not None:
                self.identity_detector = torch.compile(
                    self.identity_detector, **compile_kwargs
                )
            # Note: img2pose (FasterDoFRCNN) uses GeneralizedRCNN which has
            # dynamic control flow incompatible with torch.compile in
            # reduce-overhead mode. Skipped for now.

    def __repr__(self):
        return f"Detector(face_model={self.info['face_model']}, landmark_model={self.info['landmark_model']}, au_model={self.info['au_model']}, emotion_model={self.info['emotion_model']}, facepose_model={self.info['facepose_model']}, identity_model={self.info['identity_model']})"

    @torch.inference_mode()
    def detect_faces(self, images, face_size=112, face_detection_threshold=0.5):
        """
        detect faces and poses in a batch of images using img2pose

        Args:
            img (torch.Tensor): Tensor of shape (B, C, H, W) representing the images
            face_size (int): Output size to resize face after cropping.

        Returns:
            Fex: Prediction results dataframe
        """

        # img2pose
        frames = convert_image_to_tensor(images, img_type="float32") / 255.0
        frames = frames.to(self.device)

        batch_results = []
        use_autocast = self._half_precision and str(self.device).startswith("cuda")
        for i in range(frames.size(0)):
            single_frame = frames[i, ...].unsqueeze(0)  # Extract single image from batch
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    img2pose_output = self.facepose_detector(single_frame)
            else:
                img2pose_output = self.facepose_detector(single_frame)
            img2pose_output = postprocess_img2pose(
                img2pose_output[0], detection_threshold=face_detection_threshold
            )
            bbox = img2pose_output["boxes"]
            poses = img2pose_output["dofs"]
            facescores = img2pose_output["scores"]

            # Extract faces from bbox
            if bbox.numel() != 0:
                extracted_faces, new_bbox = extract_face_from_bbox_torch(
                    single_frame, bbox, face_size=face_size
                )
            else:  # No Face Detected - use nans
                extracted_faces = torch.zeros((1, 3, face_size, face_size), device=self.device)
                bbox = torch.full((1, 4), float("nan"), device=self.device)
                new_bbox = torch.full((1, 4), float("nan"), device=self.device)
                facescores = torch.zeros((1), device=self.device)
                poses = torch.full((1, 6), float("nan"), device=self.device)

            frame_results = {
                "face_id": i,
                "faces": extracted_faces,
                "boxes": bbox,
                "new_boxes": new_bbox,
                "poses": poses,
                "scores": facescores,
            }

            # Extract Faces separately for Resmasknet
            if self.info["emotion_model"] == "resmasknet":
                if torch.all(torch.isnan(bbox)):  # No Face Detected
                    frame_results["resmasknet_faces"] = torch.full(
                        (1, 3, 224, 224), float("nan"), device=self.device
                    )
                else:
                    resmasknet_faces, _ = extract_face_from_bbox_torch(
                        single_frame, bbox, expand_bbox=1.1, face_size=224
                    )
                    frame_results["resmasknet_faces"] = resmasknet_faces

            batch_results.append(frame_results)

        return batch_results

    @torch.inference_mode()
    def forward(self, faces_data):
        """
        Run Model Inference on detected faces.

        Args:
            faces_data (list of dict): Detected faces and associated data from `detect_faces`.

        Returns:
            Fex: Prediction results dataframe
        """

        extracted_faces = torch.cat([face["faces"] for face in faces_data], dim=0)
        extracted_faces = extracted_faces.to(self.device)
        new_bboxes = torch.cat([face["new_boxes"] for face in faces_data], dim=0)
        n_faces = extracted_faces.shape[0]

        use_autocast = self._half_precision and str(self.device).startswith("cuda")

        if self.landmark_detector is not None:
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    if self.info["landmark_model"].lower() == "mobilenet":
                        landmark_input = Compose(
                            [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                        )(extracted_faces)
                        landmarks = self.landmark_detector.forward(landmark_input)
                    elif self.info["landmark_model"].lower() == "mobilefacenet":
                        landmarks = self.landmark_detector.forward(extracted_faces)[0]
                    else:
                        landmarks = self.landmark_detector.forward(extracted_faces)
            else:
                if self.info["landmark_model"].lower() == "mobilenet":
                    landmark_input = Compose(
                        [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                    )(extracted_faces)
                    landmarks = self.landmark_detector.forward(landmark_input)
                elif self.info["landmark_model"].lower() == "mobilefacenet":
                    landmarks = self.landmark_detector.forward(extracted_faces)[0]
                else:
                    landmarks = self.landmark_detector.forward(extracted_faces)
            new_landmarks = inverse_transform_landmarks_torch(landmarks.float(), new_bboxes)
        else:
            new_landmarks = torch.full((n_faces, 136), float("nan"))

        # Pre-compute HOG features once if needed by emotion SVM or AU detector
        cached_hog_features = None
        cached_hog_landmarks = None
        need_hog = (
            (self.emotion_detector is not None and self.info["emotion_model"] == "svm")
            or self.au_detector is not None
        )
        if need_hog and self.landmark_detector is not None:
            # HOG uses sklearn/scipy which requires CPU tensors
            cached_hog_features, cached_hog_landmarks = extract_hog_features(
                extracted_faces.cpu(), landmarks.cpu()
            )

        if self.emotion_detector is not None:
            if self.info["emotion_model"] == "resmasknet":
                resmasknet_faces = torch.cat(
                    [face["resmasknet_faces"] for face in faces_data], dim=0
                ).to(self.device)
                if use_autocast:
                    with torch.autocast("cuda", dtype=torch.float16):
                        emotions = self.emotion_detector.forward(resmasknet_faces)
                else:
                    emotions = self.emotion_detector.forward(resmasknet_faces)
                emotions = torch.softmax(emotions.float(), 1)
            elif self.info["emotion_model"] == "svm":
                emotions = self.emotion_detector.detect_emo(
                    frame=cached_hog_features, landmarks=[cached_hog_landmarks]
                )
                emotions = torch.tensor(emotions)
        else:
            emotions = torch.full((n_faces, 7), float("nan"))

        if self.identity_detector is not None:
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    identity_embeddings = self.identity_detector.forward(extracted_faces)
            else:
                identity_embeddings = self.identity_detector.forward(extracted_faces)
            identity_embeddings = identity_embeddings.float()
        else:
            identity_embeddings = torch.full((n_faces, 512), float("nan"))

        if self.au_detector is not None:
            aus = self.au_detector.detect_au(
                frame=cached_hog_features, landmarks=[cached_hog_landmarks]
            )
        else:
            aus = torch.full((n_faces, 20), float("nan"))

        # Create Fex Output Representation
        bboxes = torch.cat(
            [
                convert_bbox_output(
                    face_output["new_boxes"],
                    face_output["scores"],
                )
                for face_output in faces_data
            ],
            dim=0,
        )
        feat_faceboxes = pd.DataFrame(
            bboxes.cpu().detach().numpy(),
            columns=FEAT_FACEBOX_COLUMNS,
        )

        poses = torch.cat(
            [face_output["poses"] for face_output in faces_data], dim=0
        )
        feat_poses = pd.DataFrame(
            poses.cpu().detach().numpy(), columns=FEAT_FACEPOSE_COLUMNS_6D
        )

        reshape_landmarks = new_landmarks.reshape(new_landmarks.shape[0], 68, 2)
        reordered_landmarks = torch.cat(
            [reshape_landmarks[:, :, 0], reshape_landmarks[:, :, 1]], dim=1
        )
        feat_landmarks = pd.DataFrame(
            reordered_landmarks.cpu().detach().numpy(),
            columns=openface_2d_landmark_columns,
        )

        feat_aus = pd.DataFrame(aus, columns=AU_LANDMARK_MAP["Feat"])

        feat_emotions = pd.DataFrame(
            emotions.cpu().detach().numpy(), columns=FEAT_EMOTION_COLUMNS
        )

        feat_identities = pd.DataFrame(
            identity_embeddings.cpu().detach().numpy(), columns=FEAT_IDENTITY_COLUMNS[1:]
        )

        return Fex(
            pd.concat(
                [
                    feat_faceboxes,
                    feat_landmarks,
                    feat_poses,
                    feat_aus,
                    feat_emotions,
                    feat_identities,
                ],
                axis=1,
            ),
            au_columns=AU_LANDMARK_MAP["Feat"],
            emotion_columns=FEAT_EMOTION_COLUMNS,
            facebox_columns=FEAT_FACEBOX_COLUMNS,
            landmark_columns=openface_2d_landmark_columns,
            facepose_columns=FEAT_FACEPOSE_COLUMNS_6D,
            identity_columns=FEAT_IDENTITY_COLUMNS[1:],
            detector="Feat",
            face_model=self.info["face_model"],
            landmark_model=self.info["landmark_model"],
            au_model=self.info["au_model"],
            emotion_model=self.info["emotion_model"],
            facepose_model=self.info["facepose_model"],
            identity_model=self.info["identity_model"],
        )

    @torch.inference_mode()
    def forward_raw(self, faces_data):
        """Run model inference returning raw numpy arrays instead of Fex DataFrame.

        This is ~5x faster than forward() by skipping all pandas DataFrame
        construction. Used by the streaming pipeline for real-time performance.

        Returns:
            dict with keys: bboxes, landmarks, poses, aus, emotions, identities
            Each value is a numpy array of shape (n_faces, ...).
        """

        extracted_faces = torch.cat([face["faces"] for face in faces_data], dim=0)
        extracted_faces = extracted_faces.to(self.device)
        new_bboxes = torch.cat([face["new_boxes"] for face in faces_data], dim=0)
        n_faces = extracted_faces.shape[0]

        use_autocast = self._half_precision and str(self.device).startswith("cuda")

        # Landmarks
        if self.landmark_detector is not None:
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    if self.info["landmark_model"].lower() == "mobilenet":
                        landmark_input = Compose(
                            [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                        )(extracted_faces)
                        landmarks = self.landmark_detector.forward(landmark_input)
                    elif self.info["landmark_model"].lower() == "mobilefacenet":
                        landmarks = self.landmark_detector.forward(extracted_faces)[0]
                    else:
                        landmarks = self.landmark_detector.forward(extracted_faces)
            else:
                if self.info["landmark_model"].lower() == "mobilenet":
                    landmark_input = Compose(
                        [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                    )(extracted_faces)
                    landmarks = self.landmark_detector.forward(landmark_input)
                elif self.info["landmark_model"].lower() == "mobilefacenet":
                    landmarks = self.landmark_detector.forward(extracted_faces)[0]
                else:
                    landmarks = self.landmark_detector.forward(extracted_faces)
            new_landmarks = inverse_transform_landmarks_torch(landmarks.float(), new_bboxes)
        else:
            new_landmarks = torch.full((n_faces, 136), float("nan"))
            landmarks = new_landmarks

        # HOG (CPU-bound, compute once)
        cached_hog_features = None
        cached_hog_landmarks = None
        need_hog = (
            (self.emotion_detector is not None and self.info["emotion_model"] == "svm")
            or self.au_detector is not None
        )
        if need_hog and self.landmark_detector is not None:
            cached_hog_features, cached_hog_landmarks = extract_hog_features(
                extracted_faces.cpu(), landmarks.cpu()
            )

        # Emotions
        if self.emotion_detector is not None:
            if self.info["emotion_model"] == "resmasknet":
                resmasknet_faces = torch.cat(
                    [face["resmasknet_faces"] for face in faces_data], dim=0
                ).to(self.device)
                if use_autocast:
                    with torch.autocast("cuda", dtype=torch.float16):
                        emotions = self.emotion_detector.forward(resmasknet_faces)
                else:
                    emotions = self.emotion_detector.forward(resmasknet_faces)
                emotions = torch.softmax(emotions.float(), 1)
            elif self.info["emotion_model"] == "svm":
                emotions = self.emotion_detector.detect_emo(
                    frame=cached_hog_features, landmarks=[cached_hog_landmarks]
                )
                emotions = torch.tensor(emotions)
        else:
            emotions = torch.full((n_faces, 7), float("nan"))

        # Identity
        if self.identity_detector is not None:
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    identity_embeddings = self.identity_detector.forward(extracted_faces)
            else:
                identity_embeddings = self.identity_detector.forward(extracted_faces)
            identity_embeddings = identity_embeddings.float()
        else:
            identity_embeddings = torch.full((n_faces, 512), float("nan"))

        # AUs
        if self.au_detector is not None:
            aus = self.au_detector.detect_au(
                frame=cached_hog_features, landmarks=[cached_hog_landmarks]
            )
        else:
            aus = torch.full((n_faces, 20), float("nan"))

        # Build raw numpy results — no DataFrame overhead
        bboxes = torch.cat(
            [convert_bbox_output(f["new_boxes"], f["scores"]) for f in faces_data], dim=0
        )
        poses = torch.cat([f["poses"] for f in faces_data], dim=0)
        reshape_lm = new_landmarks.reshape(n_faces, 68, 2)
        reordered_lm = torch.cat([reshape_lm[:, :, 0], reshape_lm[:, :, 1]], dim=1)

        return {
            "bboxes": bboxes.cpu().numpy().astype(np.float32),
            "landmarks": reordered_lm.cpu().numpy().astype(np.float32),
            "poses": poses.cpu().numpy().astype(np.float32),
            "aus": np.asarray(aus, dtype=np.float32) if not isinstance(aus, torch.Tensor) else aus.cpu().numpy().astype(np.float32),
            "emotions": emotions.cpu().numpy().astype(np.float32) if isinstance(emotions, torch.Tensor) else np.asarray(emotions, dtype=np.float32),
            "identities": identity_embeddings.cpu().numpy().astype(np.float32),
        }

    def detect(
        self,
        inputs,
        data_type="image",
        output_size=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        face_identity_threshold=0.8,
        face_detection_threshold=0.5,
        skip_frames=None,
        progress_bar=True,
        save=None,
        **kwargs,
    ):
        """
        Detects FEX from one or more imagathe files.

        Args:
            inputs (list of str, torch.Tensor): Path to a list of paths to image files or torch.Tensor of images (B, C, H, W)
            data_type (str): type of data to be processed; Default 'image' ['image', 'tensor', 'video']
            output_size (int): image size to rescale all image preserving aspect ratio.
            batch_size (int): how many batches of images you want to run at one shot.
            num_workers (int): how many subprocesses to use for data loading.
            pin_memory (bool): If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them.
            face_identity_threshold (float): value between 0-1 to determine similarity of person using face identity embeddings; Default >= 0.8
            face_detection_threshold (float): value between 0-1 to determine if a face was detected; Default >= 0.5
            skip_frames (int or None): number of frames to skip to speed up inference (video only); Default None
            progress_bar (bool): Whether to show the tqdm progress bar. Default is True.
            **kwargs: additional detector-specific kwargs
            save (None or str or Path): if immediately append detections to a csv file at with the given name after processing each batch, which can be useful to interrupted/resuming jobs and saving memory/RAM

        Returns:
            pd.DataFrame: Concatenated results for all images in the batch
        """

        save = Path(save) if save else None

        if data_type.lower() == "image":
            data_loader = DataLoader(
                ImageDataset(
                    inputs,
                    output_size=output_size,
                    preserve_aspect_ratio=True,
                    padding=True,
                ),
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=False,
            )
        elif data_type.lower() == "tensor":
            data_loader = DataLoader(
                TensorDataset(inputs),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        elif data_type.lower() == "video":
            dataset = VideoDataset(
                inputs,
                skip_frames=skip_frames,
                output_size=output_size,
            )
            data_loader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=False,
            )

        data_iterator = tqdm(data_loader) if progress_bar else data_loader

        batch_output = []
        frame_counter = 0

        try:
            _ = next(enumerate(tqdm(data_loader)))
        except RuntimeError as e:
            raise ValueError(
                f"When using `batch_size > 1`, all images must either have the same dimension or `output_size` should be something other than `None` to pad images prior to processing\n{e}"
            )

        for batch_id, batch_data in enumerate(data_iterator):
            faces_data = self.detect_faces(
                batch_data["Image"],
                face_size=self.face_size if hasattr(self, "face_size") else 112,
                face_detection_threshold=face_detection_threshold,
            )
            batch_results = self.forward(faces_data)

            # Create metadata for each frame
            file_names = []
            frame_ids = []
            for i, face in enumerate(faces_data):
                n_faces = len(face["scores"])
                if data_type.lower() == "video":
                    current_frame_id = batch_data["Frame"].detach().numpy()[i]
                else:
                    current_frame_id = frame_counter + i
                frame_ids.append(np.repeat(current_frame_id, n_faces))
                file_names.append(np.repeat(batch_data["FileName"][i], n_faces))
            batch_results["input"] = np.concatenate(file_names)
            batch_results["frame"] = np.concatenate(frame_ids)

            # Invert the face boxes and landmarks based on the padded output size
            # Vectorized: map each row to its batch-local index, then broadcast
            unique_frames = batch_results["frame"].unique()
            frame_to_j = {frame_idx: j for j, frame_idx in enumerate(unique_frames)}
            row_j = batch_results["frame"].map(frame_to_j).values

            scales = batch_data["Scale"].detach().numpy()[row_j]
            pad_left = batch_data["Padding"]["Left"].detach().numpy()[row_j]
            pad_top = batch_data["Padding"]["Top"].detach().numpy()[row_j]

            orig_sizes = compute_original_image_size(batch_data).numpy()
            batch_results["FrameHeight"] = orig_sizes[row_j, 0]
            batch_results["FrameWidth"] = orig_sizes[row_j, 1]

            batch_results["FaceRectX"] = (batch_results["FaceRectX"].values - pad_left) / scales
            batch_results["FaceRectY"] = (batch_results["FaceRectY"].values - pad_top) / scales
            batch_results["FaceRectWidth"] = batch_results["FaceRectWidth"].values / scales
            batch_results["FaceRectHeight"] = batch_results["FaceRectHeight"].values / scales

            x_cols = [f"x_{i}" for i in range(68)]
            y_cols = [f"y_{i}" for i in range(68)]
            batch_results[x_cols] = (batch_results[x_cols].values - pad_left[:, None]) / scales[:, None]
            batch_results[y_cols] = (batch_results[y_cols].values - pad_top[:, None]) / scales[:, None]

            if save:
                batch_results.to_csv(save, mode="a", index=False, header=batch_id == 0)
            else:
                batch_output.append(batch_results)
            frame_counter += 1 * batch_size

        batch_output = (
            Fex(
                pd.read_csv(save),
                au_columns=AU_LANDMARK_MAP["Feat"],
                emotion_columns=FEAT_EMOTION_COLUMNS,
                facebox_columns=FEAT_FACEBOX_COLUMNS,
                landmark_columns=openface_2d_landmark_columns,
                facepose_columns=FEAT_FACEPOSE_COLUMNS_6D,
                identity_columns=FEAT_IDENTITY_COLUMNS[1:],
                detector="Feat",
                face_model=self.info["face_model"],
                landmark_model=self.info["landmark_model"],
                au_model=self.info["au_model"],
                emotion_model=self.info["emotion_model"],
                facepose_model=self.info["facepose_model"],
                identity_model=self.info["identity_model"],
            )
            if save
            else pd.concat(batch_output).reset_index(drop=True)
        )
        if data_type.lower() == "video":
            batch_output["approx_time"] = [
                dataset.calc_approx_frame_time(x)
                for x in batch_output["frame"].to_numpy()
            ]
        batch_output.compute_identities(threshold=face_identity_threshold, inplace=True)
        # Overwrite with approx_time and identity columns
        if save:
            batch_output.to_csv(save, mode="w", index=False)
        return batch_output

    @torch.inference_mode()
    def detect_stream(
        self,
        source,
        batch_size=1,
        max_fps=None,
        skip_frames=None,
        face_detection_threshold=0.5,
        max_queue_size=60,
        callback=None,
    ):
        """Process video frames as a stream, yielding results in real-time.

        Uses threaded frame decoding to overlap I/O with model inference.
        Results are lightweight StreamingResult objects instead of Fex DataFrames.

        Args:
            source: Video file path (str) or webcam device index (int).
            batch_size (int): Number of frames to batch for inference.
            max_fps (float): Target FPS cap. None = no cap.
            skip_frames (int): Fixed frame skip interval. None = process every frame.
            face_detection_threshold (float): Minimum face detection confidence.
            max_queue_size (int): Frame buffer size for decode thread.
            callback (callable): Optional callback(StreamingResult) called per frame.

        Yields:
            StreamingResult: Detection results for each processed frame.

        Example:
            >>> detector = Detector(device="cuda")
            >>> for result in detector.detect_stream("video.mp4", batch_size=4):
            ...     for face in result.faces:
            ...         print(f"Frame {result.frame_idx}: emotion={face.emotions}")
        """

        video_source = StreamingVideoSource(
            source=source,
            max_queue_size=max_queue_size,
            skip_frames=skip_frames,
            target_fps=max_fps,
            output_tensor=True,
        )

        with video_source:
            batch_frames = []
            batch_meta = []  # (frame_idx, timestamp)

            for frame_idx, timestamp, frame_tensor in video_source:
                batch_frames.append(frame_tensor)
                batch_meta.append((frame_idx, timestamp))

                if len(batch_frames) >= batch_size:
                    yield from self._process_stream_batch(
                        batch_frames, batch_meta, face_detection_threshold, callback
                    )
                    batch_frames = []
                    batch_meta = []

            # Process remaining frames
            if batch_frames:
                yield from self._process_stream_batch(
                    batch_frames, batch_meta, face_detection_threshold, callback
                )

    def _process_stream_batch(
        self, batch_frames, batch_meta, face_detection_threshold, callback
    ):
        """Process a batch of frames and yield StreamingResult objects."""

        # Stack frames into batch tensor
        batch_tensor = torch.stack(batch_frames)

        # Run face detection
        faces_data = self.detect_faces(
            batch_tensor,
            face_size=self.face_size if hasattr(self, "face_size") else 112,
            face_detection_threshold=face_detection_threshold,
        )

        # Use forward_raw to skip DataFrame overhead (~5x faster)
        raw = self.forward_raw(faces_data)

        # Convert raw arrays to StreamingResults, one per frame
        face_offset = 0
        for i, (frame_idx, timestamp) in enumerate(batch_meta):
            n_faces_in_frame = len(faces_data[i]["scores"])
            face_results = []

            for f in range(n_faces_in_frame):
                row = face_offset + f
                face_results.append(
                    FaceResult(
                        bbox=raw["bboxes"][row],
                        landmarks=raw["landmarks"][row],
                        pose=raw["poses"][row],
                        aus=raw["aus"][row],
                        emotions=raw["emotions"][row],
                        identity_embedding=raw["identities"][row],
                    )
                )

            result = StreamingResult(
                frame_idx=frame_idx,
                timestamp=timestamp,
                faces=face_results,
            )

            if callback is not None:
                callback(result)

            yield result
            face_offset += n_faces_in_frame

    def detect_realtime(
        self,
        device=0,
        display=True,
        max_fps=30,
        batch_size=1,
        face_detection_threshold=0.5,
        callback=None,
    ):
        """Process webcam feed in real-time with optional display.

        Convenience wrapper around detect_stream for webcam input.

        Args:
            device (int): Webcam device index (default 0).
            display (bool): Show OpenCV window with annotated faces.
            max_fps (float): Target FPS cap.
            batch_size (int): Frames to batch for inference.
            face_detection_threshold (float): Minimum face detection confidence.
            callback (callable): Optional callback(StreamingResult) per frame.

        Example:
            >>> detector = Detector(device="cuda")
            >>> detector.detect_realtime(device=0, display=True, max_fps=30)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for real-time display. "
                "Install with: pip install opencv-python"
            )

        results = []

        for result in self.detect_stream(
            source=device,
            batch_size=batch_size,
            max_fps=max_fps,
            face_detection_threshold=face_detection_threshold,
            callback=callback,
        ):
            results.append(result)

            if display:
                # We need to get the raw frame for display — read from webcam
                # For now, show a simple info overlay
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                info_text = f"Frame: {result.frame_idx} | Faces: {len(result.faces)}"
                cv2.putText(
                    frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

                for i, face in enumerate(result.faces):
                    emo_idx = np.argmax(face.emotions)
                    emo_names = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
                    emo_text = f"Face {i}: {emo_names[emo_idx]} ({face.emotions[emo_idx]:.2f})"
                    cv2.putText(
                        frame, emo_text, (10, 60 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    )

                cv2.imshow("Py-FEAT Real-Time", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    break

        if display:
            cv2.destroyAllWindows()

        return results
