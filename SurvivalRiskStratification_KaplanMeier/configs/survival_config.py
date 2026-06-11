import sys
sys.path.append("/path/to/Med_DINOv3/Finetune/Large-Scale-Medical/Downstream/monai/LUNA16/networks/dinov3")


class SurvivalConfig:
    """
    Configuration for survival analysis and risk stratification.
    """

    def __init__(self):
        # =======================================
        # Dataset paths
        # =======================================
        self.data_root = "/path/to/Med_DINOv3/Datasets/Finetune_classification/Survival/"
        self.clinical_csv = "UPENN-GBM_clinical_info_v2.1 (1).csv"
        self.image_csv = "gbm_survival_dataset/upenn_gbm_survival_dataset_converted.csv"
        self.train_csv_0 = "train_0.csv"
        self.train_csv_1 = "train_1.csv"
        self.valid_csv = "valid.csv"

        # =======================================
        # Model configuration
        # =======================================
        self.encoder_name = "meddinov3"  # Options: meddinov3, dinov3, BrainMVP, bm_mae, scratch
        self.n_slices = 128
        self.lora_rank = 8
        self.num_classes = 2  # Binary classification (low risk vs high risk)

        # =======================================
        # Training configuration
        # =======================================
        self.batch_size = 2
        self.val_batch = 4
        self.lr = 1e-4
        self.epochs = 200
        self.val_epoch = 1  # Validate every epoch
        self.patience = 10  # Early stopping patience
        self.num_workers = 4
        self.train_ratio = 1.0  # Use 100% of training data

        # =======================================
        # Survival analysis configuration
        # =======================================
        self.risk_threshold_days = 365  # 1 year threshold for binary labels

        # =======================================
        # Hardware configuration
        # =======================================
        self.gpu_ids = [0]
        self.benchmark = False

        # =======================================
        # Scheduler configuration
        # =======================================
        self.scheduler = "ReduceLROnPlateau"

        # =======================================
        # Logging configuration
        # =======================================
        self.save_root = "./SurvivalRiskStratification_KaplanMeier/results"
        self.note = "survival_risk_stratification"
        self.manualseed = 111
        self.attr = 'class'  # Required by recorder.py (not 'args')
        self.model = 'SurvivalRiskStratification'  # Required by base_trainer.py

    def display(self, logger=None):
        """Display configuration"""
        print("=" * 60)
        print("Survival Risk Stratification Configuration")
        print("=" * 60)
        for k in dir(self):
            if not k.startswith("_") and not callable(getattr(self, k)):
                value = getattr(self, k)
                print(f"{k:30s}: {value}")
        print("=" * 60)

    def update_save_path(self, encoder_name, train_ratio=1.0):
        """Update save path based on encoder name and train ratio"""
        self.encoder_name = encoder_name
        self.train_ratio = train_ratio
        self.note = f"survival_{encoder_name}"
        self.save_root = f"./SurvivalRiskStratification_KaplanMeier/results/{encoder_name}"
