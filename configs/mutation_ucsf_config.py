"""
Configuration for UCSF-PDGM Mutation Classification

Binary classification: IDH wildtype (0) vs non-wildtype (1)
Uses 2 modalities: T1c and FLAIR
"""

class MutationUCSFConfig:
    def __init__(self):
        # ---- Dataset ----
        self.dataset_name = "UCSF_Mutation"
        self.data_root = "/path/to/Med_DINOv3/Datasets/Finetune_classification/Mutation"

        # ---- Model ----
        self.encoder_name = "meddinov3"  # Options: meddinov3, dinov3, BrainMVP, bm_mae, scratch
        self.n_slices = 128
        self.lora_rank = 8
        self.model = "slice_student_multimodal_2mod"
        self.attr = 'class'  # Required by base_trainer

        # ---- Training ----
        self.batch_size = 4
        self.val_batch = 4
        self.lr = 1e-4
        self.epochs = 8
        self.val_epoch = 1
        self.patience = 4
        self.num_workers = 4
        self.train_ratio = 1.0  # Use 100% of training data

        # ---- GPU ----
        self.gpu_ids = [0]

        # ---- Scheduler ----
        self.scheduler = "ReduceLROnPlateau"

        # ---- Logging ----
        self.save_root = f"./mutation_ucsf_results/{self.encoder_name}"
        self.note = f"mutation_ucsf_{self.encoder_name}"
        self.benchmark = False

        # ---- Random Seed ----
        self.manualseed = 111

    def display(self, logger=None):
        """Display configuration"""
        print("=" * 60)
        print(f"UCSF-PDGM Mutation Classification Config")
        print("=" * 60)
        for k in dir(self):
            if not k.startswith("_") and not callable(getattr(self, k)):
                value = getattr(self, k)
                print(f"{k:20s}: {value}")
        print("=" * 60)
