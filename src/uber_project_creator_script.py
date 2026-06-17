import yaml
import os
FILES_NAMES ='''''GSM5652176_Adipocytes-Z000000T7.hg38.bigwig
GSM5652177_Adipocytes-Z000000T9.hg38.bigwig
GSM5652178_Adipocytes-Z000000T5.hg38.bigwig
GSM5652179_Aorta-Endothel-Z00000422.hg38.bigwig
GSM5652180_Aorta-Endothel-Z0000043G.hg38.bigwig
GSM5652181_Saphenous-Vein-Endothel-Z000000RM.hg38.bigwig
GSM5652182_Saphenous-Vein-Endothel-Z000000S7.hg38.bigwig
GSM5652183_Saphenous-Vein-Endothel-Z000000SB.hg38.bigwig
GSM5652184_Kidney-Glomerular-Endothel-Z000000Q5.hg38.bigwig
GSM5652185_Kidney-Glomerular-Endothel-Z00000443.hg38.bigwig
GSM5652186_Kidney-Glomerular-Endothel-Z0000045J.hg38.bigwig
GSM5652187_Kidney-Tubular-Endothel-Z000000PX.hg38.bigwig
GSM5652188_Kidney-Tubular-Endothel-Z000000Q3.hg38.bigwig
GSM5652189_Kidney-Tubular-Endothel-Z0000042R.hg38.bigwig
GSM5652190_Liver-Endothelium-Z000000RB.hg38.bigwig
GSM5652191_Lung-Alveolar-Endothel-Z000000Q1.hg38.bigwig
GSM5652192_Lung-Alveolar-Endothel-Z000000QK.hg38.bigwig
GSM5652193_Lung-Alveolar-Endothel-Z0000045H.hg38.bigwig
GSM5652194_Pancreas-Endothel-Z0000042D.hg38.bigwig
GSM5652195_Pancreas-Endothel-Z0000042X.hg38.bigwig
GSM5652196_Pancreas-Endothel-Z00000430.hg38.bigwig
GSM5652197_Pancreas-Islet-Endothel-Z0000042Y.hg38.bigwig
GSM5652198_Colon-Fibroblasts-Z0000042A.hg38.bigwig
GSM5652199_Colon-Fibroblasts-Z0000042C.hg38.bigwig
GSM5652200_Heart-Fibroblasts-Z0000043R.hg38.bigwig
GSM5652201_Heart-Fibroblasts-Z0000041V.hg38.bigwig
GSM5652202_Heart-Fibroblasts-Z0000041W.hg38.bigwig
GSM5652203_Heart-Fibroblasts-Z0000041X.hg38.bigwig
GSM5652204_Dermal-Fibroblasts-Z00000423.hg38.bigwig
GSM5652205_Skeletal-Muscle-Z00000427.hg38.bigwig
GSM5652206_Skeletal-Muscle-Z00000429.hg38.bigwig
GSM5652207_Aorta-Smooth-Muscle-Z0000041U.hg38.bigwig
GSM5652208_Coronary-Artery-Smooth-Muscle-Z00000420.hg38.bigwig
GSM5652209_Bladder-Smooth-Muscle-Z0000041Z.hg38.bigwig
GSM5652210_Prostate-Smooth-Muscle-Z0000041Y.hg38.bigwig
GSM5652211_Lung-Bronchus-Smooth-Muscle-Z00000421.hg38.bigwig
GSM5652212_Heart-Cardiomyocyte-Z0000044G.hg38.bigwig
GSM5652213_Heart-Cardiomyocyte-Z0000044K.hg38.bigwig
GSM5652214_Heart-Cardiomyocyte-Z0000044N.hg38.bigwig
GSM5652215_Heart-Cardiomyocyte-Z0000044P.hg38.bigwig
GSM5652216_Heart-Cardiomyocyte-Z0000044Q.hg38.bigwig
GSM5652217_Heart-Cardiomyocyte-Z0000044R.hg38.bigwig
GSM5652218_Bone-Osteoblasts-Z0000042Z.hg38.bigwig
GSM5652219_Oligodendrocytes-Z000000TK.hg38.bigwig
GSM5652220_Oligodendrocytes-Z0000042E.hg38.bigwig
GSM5652221_Oligodendrocytes-Z0000042L.hg38.bigwig
GSM5652222_Oligodendrocytes-Z0000042N.hg38.bigwig
GSM5652223_Cortex-Neuron-Z000000TF.hg38.bigwig
GSM5652224_Neuron-Z000000TH.hg38.bigwig
GSM5652225_Cortex-Neuron-Z0000042F.hg38.bigwig
GSM5652226_Cortex-Neuron-Z0000042H.hg38.bigwig
GSM5652227_Cortex-Neuron-Z0000042J.hg38.bigwig
GSM5652228_Cortex-Neuron-Z0000042M.hg38.bigwig
GSM5652229_Cortex-Neuron-Z0000042P.hg38.bigwig
GSM5652230_Cortex-Neuron-Z0000042K.hg38.bigwig
GSM5652231_Cerebellum-Neuron-Z000000TB.hg38.bigwig
GSM5652232_Cortex-Neuron-Z000000TD.hg38.bigwig
GSM5652233_Liver-Hepatocytes-Z000000R3.hg38.bigwig
GSM5652234_Liver-Hepatocytes-Z000000T3.hg38.bigwig
GSM5652235_Liver-Hepatocytes-Z0000043Q.hg38.bigwig
GSM5652236_Liver-Hepatocytes-Z0000044H.hg38.bigwig
GSM5652237_Liver-Hepatocytes-Z0000044M.hg38.bigwig
GSM5652238_Liver-Hepatocytes-Z00000431.hg38.bigwig
GSM5652239_Pancreas-Duct-Z0000043T.hg38.bigwig
GSM5652240_Pancreas-Duct-Z0000043U.hg38.bigwig
GSM5652241_Pancreas-Duct-Z0000043V.hg38.bigwig
GSM5652242_Pancreas-Duct-Z000000QZ.hg38.bigwig
GSM5652243_Pancreas-Acinar-Z000000QX.hg38.bigwig
GSM5652244_Pancreas-Acinar-Z0000043W.hg38.bigwig
GSM5652245_Pancreas-Acinar-Z0000043X.hg38.bigwig
GSM5652246_Pancreas-Acinar-Z0000043Y.hg38.bigwig
GSM5652247_Pancreas-Delta-Z00000451.hg38.bigwig
GSM5652248_Pancreas-Delta-Z00000454.hg38.bigwig
GSM5652249_Pancreas-Delta-Z00000457.hg38.bigwig
GSM5652250_Pancreas-Beta-Z00000452.hg38.bigwig
GSM5652251_Pancreas-Beta-Z00000455.hg38.bigwig
GSM5652252_Pancreas-Beta-Z00000458.hg38.bigwig
GSM5652253_Pancreas-Alpha-Z00000453.hg38.bigwig
GSM5652254_Pancreas-Alpha-Z00000456.hg38.bigwig
GSM5652255_Pancreas-Alpha-Z00000459.hg38.bigwig
GSM5652256_Kidney-Glomerular-Epithelial-Z0000045K.hg38.bigwig
GSM5652257_Kidney-Glomerular-Epithelial-Z0000045L.hg38.bigwig
GSM5652258_Kidney-Tubular-Epithelial-Z000000QH.hg38.bigwig
GSM5652259_Kidney-Tubular-Epithelial-Z0000043Z.hg38.bigwig
GSM5652260_Kidney-Tubular-Epithelial-Z00000440.hg38.bigwig
GSM5652261_Kidney-Glomerular-Podocytes-Z0000042W.hg38.bigwig
GSM5652262_Kidney-Glomerular-Podocytes-Z00000441.hg38.bigwig
GSM5652263_Kidney-Glomerular-Podocytes-Z00000442.hg38.bigwig
GSM5652264_Thyroid-Epithelial-Z0000042S.hg38.bigwig
GSM5652265_Thyroid-Epithelial-Z0000042T.hg38.bigwig
GSM5652266_Thyroid-Epithelial-Z0000042U.hg38.bigwig
GSM5652267_Fallopian-Epithelial-Z000000Q7.hg38.bigwig
GSM5652268_Fallopian-Epithelial-Z000000S9.hg38.bigwig
GSM5652269_Fallopian-Epithelial-Z000000UV.hg38.bigwig
GSM5652270_Ovary-Epithelial-Z000000QT.hg38.bigwig
GSM5652271_Endometrium-Epithelial-Z00000434.hg38.bigwig
GSM5652272_Endometrium-Epithelial-Z00000435.hg38.bigwig
GSM5652273_Endometrium-Epithelial-Z0000043S.hg38.bigwig
GSM5652274_Bone_marrow-Erythrocyte_progenitors-Z000000RF.hg38.bigwig
GSM5652275_Bone_marrow-Erythrocyte_progenitors-Z000000RH.hg38.bigwig
GSM5652276_Bone_marrow-Erythrocyte_progenitors-Z000000RK.hg38.bigwig
GSM5652277_Blood-T-CD3-Z000000TV.hg38.bigwig
GSM5652278_Blood-T-CD3-Z000000UP.hg38.bigwig
GSM5652279_Blood-T-CD4-Z000000TT.hg38.bigwig
GSM5652280_Blood-T-CD4-Z000000U7.hg38.bigwig
GSM5652281_Blood-T-CD4-Z000000UM.hg38.bigwig
GSM5652282_Blood-T-CD8-Z000000TR.hg38.bigwig
GSM5652283_Blood-T-CD8-Z000000U5.hg38.bigwig
GSM5652284_Blood-T-CD8-Z000000UK.hg38.bigwig
GSM5652285_Blood-T-CenMem-CD4-Z00000417.hg38.bigwig
GSM5652286_Blood-T-CenMem-CD4-Z0000041D.hg38.bigwig
GSM5652287_Blood-T-CenMem-CD4-Z0000041N.hg38.bigwig
GSM5652288_Blood-T-Eff-CD8-Z00000419.hg38.bigwig
GSM5652289_Blood-T-Eff-CD8-Z0000041F.hg38.bigwig
GSM5652290_Blood-T-Eff-CD8-Z0000041Q.hg38.bigwig
GSM5652291_Blood-T-EffMem-CD4-Z00000416.hg38.bigwig
GSM5652292_Blood-T-EffMem-CD4-Z0000041C.hg38.bigwig
GSM5652293_Blood-T-EffMem-CD4-Z0000041M.hg38.bigwig
GSM5652294_Blood-T-EffMem-CD8-Z0000041A.hg38.bigwig
GSM5652295_Blood-T-EffMem-CD8-Z0000041G.hg38.bigwig
GSM5652296_Blood-T-Naive-CD4-Z0000041E.hg38.bigwig
GSM5652297_Blood-T-Naive-CD8-Z0000041B.hg38.bigwig
GSM5652298_Blood-T-Naive-CD8-Z0000041H.hg38.bigwig
GSM5652299_Blood-NK-Z000000TM.hg38.bigwig
GSM5652300_Blood-NK-Z000000U1.hg38.bigwig
GSM5652301_Blood-NK-Z000000UF.hg38.bigwig
GSM5652302_Blood-Monocytes-Z000000TP.hg38.bigwig
GSM5652303_Blood-Monocytes-Z000000U3.hg38.bigwig
GSM5652304_Blood-Monocytes-Z000000UH.hg38.bigwig
GSM5652305_Colon-Macrophages-Z00000444.hg38.bigwig
GSM5652306_Colon-Macrophages-Z00000446.hg38.bigwig
GSM5652307_Liver-Macrophages-Z0000043P.hg38.bigwig
GSM5652308_Lung-Alveolar-Macrophages-Z00000448.hg38.bigwig
GSM5652309_Lung-Alveolar-Macrophages-Z0000044C.hg38.bigwig
GSM5652310_Lung-Interstitial-Macrophages-Z00000447.hg38.bigwig
GSM5652311_Lung-Interstitial-Macrophages-Z0000044D.hg38.bigwig
GSM5652312_Lung-Interstitial-Macrophages-Z0000044E.hg38.bigwig
GSM5652313_Blood-Granulocytes-Z000000TZ.hg38.bigwig
GSM5652314_Blood-Granulocytes-Z000000UD.hg38.bigwig
GSM5652315_Blood-Granulocytes-Z000000UT.hg38.bigwig
GSM5652316_Blood-B-Z000000TX.hg38.bigwig
GSM5652317_Blood-B-Z000000UB.hg38.bigwig
GSM5652318_Blood-B-Z000000UR.hg38.bigwig
GSM5652319_Blood-B-Mem-Z0000041J.hg38.bigwig
GSM5652320_Blood-B-Mem-Z0000041K.hg38.bigwig
GSM5652321_Epidermal-Keratinocytes-Z00000424.hg38.bigwig
GSM5652322_Tonsil-Palatine-Epithelial-Z000000QF.hg38.bigwig
GSM5652323_Tonsil-Palatine-Epithelial-Z000000RP.hg38.bigwig
GSM5652324_Tonsil-Palatine-Epithelial-Z000000RR.hg38.bigwig
GSM5652325_Tonsil-Pharyngeal-Epithelial-Z000000Q9.hg38.bigwig
GSM5652326_Tonsil-Pharyngeal-Epithelial-Z000000S1.hg38.bigwig
GSM5652327_Tongue-Epithelial-Z000000QV.hg38.bigwig
GSM5652328_Tongue-Epithelial-Z00000449.hg38.bigwig
GSM5652329_Tongue-Epithelial-Z0000044F.hg38.bigwig
GSM5652330_Tongue_base-Epithelial-Z0000044B.hg38.bigwig
GSM5652331_Larynx-Epithelial-Z000000QB.hg38.bigwig
GSM5652332_Esophagus-Epithelial-Z000000PZ.hg38.bigwig
GSM5652333_Esophagus-Epithelial-Z00000426.hg38.bigwig
GSM5652334_Pharynx-Epithelial-Z0000044A.hg38.bigwig
GSM5652335_Lung-Bronchus-Epithelial-Z000000QD.hg38.bigwig
GSM5652336_Lung-Bronchus-Epithelial-Z000000RZ.hg38.bigwig
GSM5652337_Lung-Bronchus-Epithelial-Z000000S5.hg38.bigwig
GSM5652338_Prostate-Epithelial-Z000000RV.hg38.bigwig
GSM5652339_Prostate-Epithelial-Z000000S3.hg38.bigwig
GSM5652340_Prostate-Epithelial-Z0000045F.hg38.bigwig
GSM5652341_Prostate-Epithelial-Z0000045G.hg38.bigwig
GSM5652342_Bladder-Epithelial-Z000000QM.hg38.bigwig
GSM5652343_Bladder-Epithelial-Z000000QP.hg38.bigwig
GSM5652344_Bladder-Epithelial-Z0000043F.hg38.bigwig
GSM5652345_Bladder-Epithelial-Z0000044L.hg38.bigwig
GSM5652346_Bladder-Epithelial-Z00000450.hg38.bigwig
GSM5652347_Breast-Luminal-Epithelial-Z000000V2.hg38.bigwig
GSM5652348_Breast-Luminal-Epithelial-Z000000VJ.hg38.bigwig
GSM5652349_Breast-Luminal-Epithelial-Z000000VN.hg38.bigwig
GSM5652350_Breast-Basal-Epithelial-Z000000V6.hg38.bigwig
GSM5652351_Breast-Basal-Epithelial-Z000000VG.hg38.bigwig
GSM5652352_Breast-Basal-Epithelial-Z000000VL.hg38.bigwig
GSM5652353_Breast-Basal-Epithelial-Z0000043E.hg38.bigwig
GSM5652354_Lung-Alveolar-Epithelial-Z000000T1.hg38.bigwig
GSM5652355_Lung-Alveolar-Epithelial-Z000000VC.hg38.bigwig
GSM5652356_Lung-Alveolar-Epithelial-Z000000VE.hg38.bigwig
GSM5652357_Lung-Pleura-Z0000042B.hg38.bigwig
GSM5652358_Gallbladder-Epithelial-Z00000432.hg38.bigwig
GSM5652359_Gastric-fundus-Epithelial-Z000000RX.hg38.bigwig
GSM5652360_Gastric-fundus-Epithelial-Z000000SK.hg38.bigwig
GSM5652361_Gastric-fundus-Epithelial-Z000000SV.hg38.bigwig
GSM5652362_Gastric-body-Epithelial-Z000000SD.hg38.bigwig
GSM5652363_Gastric-body-Epithelial-Z000000SM.hg38.bigwig
GSM5652364_Gastric-body-Epithelial-Z000000ST.hg38.bigwig
GSM5652365_Gastric-antrum-Epithelial-Z000000SF.hg38.bigwig
GSM5652366_Gastric-antrum-Epithelial-Z000000SP.hg38.bigwig
GSM5652367_Gastric-antrum-Epithelial-Z000000SR.hg38.bigwig
GSM5652368_Gastric-antrum-Endocrine-Z00000437.hg38.bigwig
GSM5652369_Gastric-antrum-Endocrine-Z00000438.hg38.bigwig
GSM5652370_Colon-Right-Epithelial-Z000000V0.hg38.bigwig
GSM5652371_Colon-Right-Epithelial-Z000000V8.hg38.bigwig
GSM5652372_Colon-Right-Endocrine-Z0000044S.hg38.bigwig
GSM5652373_Colon-Left-Epithelial-Z000000VA.hg38.bigwig
GSM5652374_Colon-Left-Endocrine-Z0000044J.hg38.bigwig
GSM5652375_Colon-Left-Endocrine-Z0000044T.hg38.bigwig
GSM5652376_Colon-Left-Epithelial-Z0000043B.hg38.bigwig
GSM5652377_Colon-Left-Epithelial-Z0000043C.hg38.bigwig
GSM5652378_Small-int-Epithelial-Z0000042V.hg38.bigwig
GSM5652379_Small-int-Epithelial-Z000000RT.hg38.bigwig
GSM5652380_Small-int-Epithelial-Z000000UW.hg38.bigwig
GSM5652381_Small-int-Epithelial-Z000000UY.hg38.bigwig
GSM5652382_Small-int-Endocrine-Z00000436.hg38.bigwig'''.split("\n")

TISSUE_NUM_MIN = 4
BATCH_SIZE = 64


BASE_FILE_PATH = "/shared/cycle1_huji_linial_prj/roei_test/downloaded_datasets"
BASE_PROJECT_CONFIG_PATH = "/home/users/roeizucker/tests/new_codebase/personalized_methylation_codebase/configs/kol_kora_no_var/project_configs/auto_created"
BASE_CONFIG_PATH = "/home/users/roeizucker/tests/new_codebase/personalized_methylation_codebase/configs/kol_kora_no_var"

# BASE_FILE_PATH = "/sci/archive/michall/roeizucker/downloaded_datasets"
# BASE_PROJECT_CONFIG_PATH = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/atlas_configs/auto_generated"
# BASE_CONFIG_PATH = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/atlas_configs"

DATASET_BASE_DIR = "/shared/cycle1_huji_linial_prj/roei_test/huggingface_datasets_dir"
BASE_MODEL_DIR = "/shared/cycle1_huji_linial_prj/roei_test/trained_huggingface_models_location"

# DATASET_BASE_DIR = "/sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir"
# BASE_MODEL_DIR = "/sci/labs/michall/roeizucker/trained_huggingface_models_location"

# TODO: change so it creates a kmer_split project and a window split project
def create_base_project_dict(name,grouped_files,base_file_path,filtration_method):
    base = {}
    base["params"] = {
        "project_suffix": "_kol_kora_no_var_" + name,
        "bigwig_files" : [os.path.join(base_file_path,x) for x in grouped_files],
        "names" : [x.replace(".hg38.bigwig","").split("-")[-1] for x in grouped_files],
        "created_configs_path" : os.path.join(BASE_CONFIG_PATH,"_" + name),
        "tokenizer_name"                : "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        "dataset_base_dir"              : DATASET_BASE_DIR,
        "base_model_location"           : BASE_MODEL_DIR,
        "model_type"                    : "regression_analysis",
        "train_test_seperation"         : filtration_method,
        # "chromosomes"                   : ["chr1","chr2","chr3","chr4","chr5","chr18","chr19","chr20","chr21","chr22"],
        "chromosomes"                   : ["chr5"],
        "use_lora"                      : False,
        "freeze_model"                  : False,
        "num_labels"                    : 1,
        "number_of_bins"                : 5,
        "load_best_model_at_end"        : False,
        "num_train_epoch"               : 5,
        "num_pretrain_epoch"            : 2,
        "per_device_train_batch_sizes"  : [BATCH_SIZE],
        "per_device_eval_batch_size"    : BATCH_SIZE,
        "learning_rates"                : [1e-06],
        "metric_for_best_model"         : "mse",
        "save_stratagy"                 : "steps",
        "number_of_steps"               : 10000,
        # "seq_sizes"                     : [5400,1200],
        "seq_sizes"                     : [5400],
        "test_sizes"                    : [0.2],
        "save_total_limit"              : 2,
        "add_epoch_end_save_callback"   : True,
        "save_at_end"                   : False,
        "continue_from_last"            : True,
        "use_variant_filtering"         : False,
        "override_dataset"              : False,
        "pretraining_variant_filtering_upper_bound" : -1,
        "pretraining_variant_filtering_lower_bound" : -1,
        "retraining_variant_filtering_upper_bound" : -1,
        "retraining_variant_filtering_lower_bound" : -1,
        # "pretraining_variant_filtering_upper_bound" : 0.9,
        # "pretraining_variant_filtering_lower_bound" : 0,
        # "retraining_variant_filtering_upper_bound" : 2,
        # "retraining_variant_filtering_lower_bound" : 0.9,
        "max_grad_norm"                 : 1,
        "top_rows"                      : -1,
        "min_number_of_cpg_sites"       : 10,
        "load_dataset_to_memory"        : True,
    }
    return base

grouped_files = {}
for i in range(len(FILES_NAMES)):
    a = "_".join(FILES_NAMES[i].split("_")[1:])
    suffix = ("-".join(a.split("-")[:-1]))
    if suffix == "":
        print(FILES_NAMES[i])
    if suffix not in grouped_files:
        grouped_files[suffix] = []
    grouped_files[suffix].append(FILES_NAMES[i])
    # brea
all_sum = 0
for suffix in sorted(grouped_files):
    if len(grouped_files[suffix]) < TISSUE_NUM_MIN:
        continue
    kmer_project_dict = create_base_project_dict(suffix + "_kmer",grouped_files[suffix],BASE_FILE_PATH,"kmer_sample_filtration")
    window_project_dict = create_base_project_dict(suffix + "_window",grouped_files[suffix],BASE_FILE_PATH,"random_sample_filtration")
    all_sum+=len(grouped_files[suffix])
    base_tissue_path = os.path.join(BASE_PROJECT_CONFIG_PATH,suffix)
    print(base_tissue_path)
    if not os.path.exists(base_tissue_path):
        os.mkdir(base_tissue_path)
    # Fix this
    with open(os.path.join(base_tissue_path,suffix + "_kmer.yaml"), 'w') as file:
        yaml.dump(kmer_project_dict, file, default_flow_style=False, sort_keys=False)
    with open(os.path.join(base_tissue_path,suffix + "_window.yaml"), 'w') as file:
        yaml.dump(window_project_dict, file, default_flow_style=False, sort_keys=False)
