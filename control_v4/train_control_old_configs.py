############################
# CONFIGURATION PARAMETERS #
############################

# Icons-50 1024 GBN
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/target"

OUTPUT_DIR = "control_v4/train_outputs_Icons-50_1024_GBN"
FREEZE_DENOISER = False  # NOTE
BASE_CKPT_PATH = ""

# OUTPUT_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_PRETRAINED"
# EPOCHS = 10000
# TRAIN_TRUNCATION_RATIO = 0.30
# INFER_TRUNCATION_RATIO = 0.30
# RESAMPLE_JUMPS = 2


# Icons-50 1024 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_Icons-50_1024_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""

# OUTPUT_DIR = "control_v4/train_outputs_Icons-50_1024_WVS_PRETRAINED"
# EPOCHS = 10000
# TRAIN_TRUNCATION_RATIO = 0.30
# INFER_TRUNCATION_RATIO = 0.30
# RESAMPLE_JUMPS = 2


# CelebA 1024 GBN - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_CelebA-5K_1024_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""


# CelebA 1024 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_CelebA-5K_1024_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""


# ShapeNetRendering 256 GBN - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRendering-3K_256_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRendering-3K_256_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_ShapeNetRendering-3K_256_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 16


# ShapeNetRendering 256 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRendering-3K_256_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRendering-3K_256_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_ShapeNetRendering-3K_256_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 16


# ShapeNetRenderingV2 576 GBN - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRenderingV2-3K_576_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRenderingV2-3K_576_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_ShapeNetRenderingV2-3K_576_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# ShapeNetRenderingV2 576 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRenderingV2-3K_576_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRenderingV2-3K_576_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_ShapeNetRenderingV2-3K_576_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# AirplaneCarShip 1600 GBN - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AirplaneCarShip-3K_1600_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AirplaneCarShip-3K_1600_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_AirplaneCarShip-3K_1600_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 40


# AirplaneCarShip 1600 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AirplaneCarShip-3K_1600_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AirplaneCarShip-3K_1600_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_AirplaneCarShip-3K_1600_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 40


# ShapeNetRender_Custom 1600 GBN - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_ShapeNetRender_Custom-3K_1600_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 40


# ShapeNetRender_Custom 1600 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_ShapeNetRender_Custom-3K_1600_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 40


# ShapeNetRender_Custom 576 GBN - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_576_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_576_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_ShapeNetRender_Custom-3K_576_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# ShapeNetRender_Custom 576 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_576_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_576_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_ShapeNetRender_Custom-3K_576_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# bizarre-pose-estimator 576 GBN - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/bizarre-pose-estimator_576_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/bizarre-pose-estimator_576_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_bizarre-pose-estimator_576_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# bizarre-pose-estimator 576 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/bizarre-pose-estimator_576_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/bizarre-pose-estimator_576_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_bizarre-pose-estimator_576_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# fg-01 576 GBN - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/fg-01_576_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/fg-01_576_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_fg-01_576_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# fg-01 576 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/fg-01_576_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/fg-01_576_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_fg-01_576_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# Disney-Characters 576 GBN - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Disney-Characters_576_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Disney-Characters_576_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_Disney-Characters_576_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# Disney-Characters 576 WVS - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Disney-Characters_576_WVS/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Disney-Characters_576_WVS/target"

# OUTPUT_DIR = "control_v4/train_outputs_Disney-Characters_576_WVS"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
# GRID_SIZE = 24


# ANIMALS 1024 GBN
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AM-2K_1024_GBN/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AM-2K_1024_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_AM-2K_1024_GBN"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""


# Stress 1 1024 - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/target"
# VAL_SPLIT = 0.0              # NOTE

# OUTPUT_DIR = "control_v4/train_outputs_data_stress1"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""


# Stress 2 1024 V2 - NOTE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2_V2/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2_V2/target"
# VAL_SPLIT = 0.0              # NOTE

# OUTPUT_DIR = "control_v4/train_outputs_data_stress2_V2"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""


# Stress 2 1024 V1 - IGNORE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2/target"
# VAL_SPLIT = 0.0              # NOTE

# OUTPUT_DIR = "control_v4/train_outputs_data_stress2"
# FREEZE_DENOISER = False  # NOTE
# BASE_CKPT_PATH = ""
