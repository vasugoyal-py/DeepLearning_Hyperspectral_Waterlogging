"""
Central configuration for DeepWaterLog — Deep Learning–Enabled Hyperspectral
Remote Sensing for Agricultural Waterlogging Detection.

Every physical, spectral, statistical and deep-learning parameter used by the
pipeline is declared here so that a run can be reproduced exactly.  Values are
grouped by pipeline stage and include the literature source that motivated them.

Theme: Smart AgriTech — Precision Farming and Agro Sustainability

References
----------
- Thenkabail et al. (2018). Hyperspectral Remote Sensing of Vegetation.
- Crawford & Srinivasan (2016). Hyperspectral Remote Sensing.
- Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical
  Image Segmentation.
- Krizhevsky et al. (2012). ImageNet Classification with Deep CNNs.
- APA 7th Edition (2020). Publication Manual of the American Psychological
  Association.
"""

from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
PROJECT_NAME = "DeepWaterLog — Deep Learning–Enabled Hyperspectral Remote Sensing for Agricultural Waterlogging Detection"
PROJECT_VERSION = "1.0.0"
PROJECT_SHORT = "DeepWaterLog"
THEME = "Smart AgriTech — Precision Farming and Agro Sustainability"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HYPERSPECTRAL_DIR = DATA_DIR / "hyperspectral"
LABELS_DIR = DATA_DIR / "labels"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
RESULTS_DIR = DATA_DIR / "results"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"
CRED_FILE = BASE_DIR / "do_not_commit" / "cred.json"
SCENARIOS_DB = DATA_DIR / "scenarios.json"

# ---------------------------------------------------------------------------
# Hyperspectral preprocessing defaults (Stage 1)
# ---------------------------------------------------------------------------
HYPER_SPECTRAL_BANDS_DEFAULT = 224          # typical number of bands in an AVIRIS-style cube
BAD_BAND_REGIONS_NM = [(1340, 1450), (1780, 2000)]  # water-absorption bands to remove
DESTRIPE_METHOD = "wavelet"               # "wavelet", "polynomial", or "none"
NOISE_FILTER = "savitzky_golay"           # "savitzky_golay", "wavelet", or "none"
ATMOSPHERIC_CORRECTION = "FLAASH"         # "FLAASH", "QUAC", or "none"
RADIOMETRIC_CALIBRATION = True            # convert DN to radiance/reflectance
OUTPUT_REFLECTANCE = True                 # produce surface-reflectance cube

# ---------------------------------------------------------------------------
# Band selection & dimensionality reduction defaults (Stage 2)
# ---------------------------------------------------------------------------
BAND_SELECTION_METHOD = "mutual_information"   # "correlation", "mutual_information", "pca", "none"
CORRELATION_THRESHOLD = 0.95          # remove bands with pairwise correlation above this
MI_TOP_K_BANDS = 30                   # number of top bands to retain by mutual information
PCA_COMPONENTS = 30                   # number of principal components if PCA is used
WAVELENGTH_REGIONS_OF_INTEREST = {
    "visible": (400, 700),            # chlorophyll and pigment absorption
    "red_edge": (700, 750),           # vegetation stress indicator
    "nir": (750, 900),                # canopy structure and water content
    "swir1": (1550, 1750),            # soil and crop moisture
    "swir2": (2080, 2350),            # dry matter and lignin
}

# ---------------------------------------------------------------------------
# Training data preparation defaults (Stage 3)
# ---------------------------------------------------------------------------
PATCH_SIZE = 64                       # pixels per side for CNN/U-Net patches
STRIDE = 32                           # patch extraction stride (overlap = PATCH_SIZE - STRIDE)
CLASSES = [
    "waterlogged_soil",
    "healthy_crop",
    "stressed_crop",
    "open_water",
    "non_agricultural",
]
CLASS_COLORS = {
    "waterlogged_soil": (255, 0, 0),
    "healthy_crop": (0, 255, 0),
    "stressed_crop": (255, 165, 0),
    "open_water": (0, 0, 255),
    "non_agricultural": (128, 128, 128),
}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NORMALIZATION = "min_max"             # "min_max", "z_score", or "none"
AUGMENTATION = True                   # horizontal/vertical flip, rotation, spectral jitter

# ---------------------------------------------------------------------------
# Deep learning model defaults (Stage 4)
# ---------------------------------------------------------------------------
# CNN (patch-level classification)
CNN_ARCHITECTURE = "custom_cnn"       # "custom_cnn", "resnet18", "resnet50", "efficientnet_b0"
CNN_INPUT_CHANNELS = 30               # matches PCA components or selected band count
CNN_NUM_CLASSES = len(CLASSES)
CNN_LEARNING_RATE = 1e-4
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 100
CNN_EARLY_STOPPING_PATIENCE = 15
CNN_OPTIMIZER = "adam"                # "adam", "sgd", "adamw"
CNN_LOSS = "cross_entropy"            # "cross_entropy", "focal", "weighted_cross_entropy"

# U-Net (pixel-level semantic segmentation)
UNET_ARCHITECTURE = "unet"            # "unet", "unet_plusplus", "attention_unet"
UNET_INPUT_CHANNELS = 30
UNET_NUM_CLASSES = len(CLASSES)
UNET_LEARNING_RATE = 1e-4
UNET_BATCH_SIZE = 16
UNET_EPOCHS = 100
UNET_EARLY_STOPPING_PATIENCE = 15
UNET_OPTIMIZER = "adam"
UNET_LOSS = "cross_entropy"           # "cross_entropy", "dice", "focal", "combined"
UNET_DEPTH = 4                        # number of encoder/decoder levels
UNET_BASE_FILTERS = 64                # filters at the first level

# ---------------------------------------------------------------------------
# Evaluation defaults (Stage 5)
# ---------------------------------------------------------------------------
EVAL_METRICS = ["accuracy", "precision", "recall", "f1", "iou"]
IOU_THRESHOLD = 0.5                   # threshold for binarizing soft predictions
CLASS_WEIGHTS = "balanced"            # "balanced", "inverse_frequency", or "none"
CONFUSION_MATRIX_NORMALIZE = "true"   # "true", "pred", "all", or "none"

# ---------------------------------------------------------------------------
# Mapping & decision support defaults (Stage 6)
# ---------------------------------------------------------------------------
MAP_OUTPUT_FORMAT = "geotiff"         # "geotiff", "png", or "both"
MAP_RESOLUTION_M = 10                 # output map spatial resolution in metres
UI_PORT = 5000                        # Flask development server port
UI_MAP_TILES = "OpenStreetMap"        # base map tile source

# ---------------------------------------------------------------------------
# Run artefacts
# ---------------------------------------------------------------------------
SAVE_RESULTS = True
RUN_LOG_JSONL = "run_log.jsonl"
SAVE_MODEL_CHECKPOINTS = True
BEST_MODEL_METRIC = "f1"              # metric used to select the best checkpoint

# ---------------------------------------------------------------------------
# Quality control thresholds
# ---------------------------------------------------------------------------
QC_MIN_CLASS_FREQUENCY = 0.01         # classes below 1% of pixels trigger a warning
QC_MAX_NO_DATA_FRACTION = 0.10        # patches with >10% NaN are excluded
QC_MIN_TRAIN_ACCURACY = 0.70          # training accuracy below this is a warning
QC_MIN_VAL_F1 = 0.60                  # validation F1 below this is a warning

# ---------------------------------------------------------------------------
# Convenience snapshot used in provenance records
# ---------------------------------------------------------------------------
def config_snapshot():
    """Return a JSON-serialisable snapshot of the scientific parameters."""
    keys = [
        "PROJECT_NAME", "PROJECT_VERSION", "THEME",
        "HYPER_SPECTRAL_BANDS_DEFAULT", "BAD_BAND_REGIONS_NM", "DESTRIPE_METHOD",
        "NOISE_FILTER", "ATMOSPHERIC_CORRECTION", "RADIOMETRIC_CALIBRATION",
        "OUTPUT_REFLECTANCE", "BAND_SELECTION_METHOD", "CORRELATION_THRESHOLD",
        "MI_TOP_K_BANDS", "PCA_COMPONENTS", "WAVELENGTH_REGIONS_OF_INTEREST",
        "PATCH_SIZE", "STRIDE", "CLASSES", "CLASS_COLORS",
        "TRAIN_RATIO", "VAL_RATIO", "TEST_RATIO", "NORMALIZATION", "AUGMENTATION",
        "CNN_ARCHITECTURE", "CNN_INPUT_CHANNELS", "CNN_NUM_CLASSES",
        "CNN_LEARNING_RATE", "CNN_BATCH_SIZE", "CNN_EPOCHS",
        "CNN_EARLY_STOPPING_PATIENCE", "CNN_OPTIMIZER", "CNN_LOSS",
        "UNET_ARCHITECTURE", "UNET_INPUT_CHANNELS", "UNET_NUM_CLASSES",
        "UNET_LEARNING_RATE", "UNET_BATCH_SIZE", "UNET_EPOCHS",
        "UNET_EARLY_STOPPING_PATIENCE", "UNET_OPTIMIZER", "UNET_LOSS",
        "UNET_DEPTH", "UNET_BASE_FILTERS",
        "EVAL_METRICS", "IOU_THRESHOLD", "CLASS_WEIGHTS",
        "CONFUSION_MATRIX_NORMALIZE", "MAP_OUTPUT_FORMAT", "MAP_RESOLUTION_M",
        "UI_PORT", "UI_MAP_TILES",
        "SAVE_RESULTS", "SAVE_MODEL_CHECKPOINTS", "BEST_MODEL_METRIC",
        "QC_MIN_CLASS_FREQUENCY", "QC_MAX_NO_DATA_FRACTION",
        "QC_MIN_TRAIN_ACCURACY", "QC_MIN_VAL_F1",
    ]
    snapshot = {}
    for key in keys:
        value = globals().get(key)
        if isinstance(value, tuple):
            value = list(value)
        snapshot[key] = deepcopy(value)
    return snapshot
