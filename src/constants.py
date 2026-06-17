CPG_TRAINING_TASK_TYPE = "cpg_training"
CPG_RETRAINING_TASK_TYPE = "cpg_retraining"
CHECKPINT_SAVE_PREFIX = "checkpoint-"
CHECKPINT_SAVE_SEPERATOR = "-"
DEFAULT_RANDOM_STATE = 42

NO_PRETRAINING_CONFIG_NAME = "no_pretraining"

SAVED_EPOCH_PREFIX = "epoch"


REGRESSION_ANALYSIS_SYMBOL = "regression_analysis"
CLASSIFICATION_ANALYSIS_SYMBOL = "classification_analysis"

# formatting constants:
STD_VARIABILITY_TYPE = "std"
QUANTILE_SEPERATION_TYPE = "quantile"



# data extractor constants
BLANK_LABEL_VALUE = -100
SINGLE_INSTANCE_FORMAT_VALUE_NAME = "value"
HG38_ENCODING = "HG38"
HG38_PATH = "/sci/archive/michall/roeizucker/reference_genome/hg38.fa"
RAW_INPUT_NAME = "raw_input"
PREPROCESSED_INPUT_NAME = "pre_processed_input"
INTERMEDIATE_INPUT_NAME = "intermediate_input"
DEFAULT_FULL_POSITION_COLUMN_NAME = "full_position"
CPG_EXTRACTION_TASK_NAME = "cpg_extraction"
CPG_SEPERATING_SITES_TASK_NAME = "cpg_seperating_sites"
CPG_TOKEN_CLASSIFICATION_EXTRACTION_TASK_NAME = "cpg_token_classification_extraction"

STANDART_NO_OVERLAP_WINDOW_TYPE = "standart_no_overlap"
WINDOW_NAME_TRAIN_TEST_FILTRATION = "window_name_filtration"
RANDOM_SAMPLE_TRAIN_TEST_FILTRATION = "random_sample_filtration"
KMER_SAMPLE_TRAIN_TEST_FILTRATION = "kmer_sample_filtration"
    # encode specific constantes
ENCODE_VALUE_NAME = "methyl_rate"
INSTADEEP_KMER_SIZE = 6
DATASET_BATCH_SIZE = 50



CPG_EVALUATION_TASK_TYPE = "cpg_evaluation"
EVALUATE_MULTIPLE_CHCEKPOINTS_SUBTASK_TYPE = "evaluate_multiple_checkpoints"
BINS_GROUPING_METHOD = "bins"