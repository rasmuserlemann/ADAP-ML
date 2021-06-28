# Author: Xiuxia Du
# April 2020

from collections import namedtuple
import os

# =================================
# file location
# =================================
bool_windows = True

batch_index_str = '42'
batch_index_int = int(batch_index_str)

# batches_to_process = ['batch-01', 'batch-02', 'batch-03', 'batch-04', 'batch-05', 'batch-06', 'batch-07', \
#                       'batch-08', 'batch-09', 'batch-10', 'batch-11', 'batch-12', 'batch-13', 'batch-14', \
#                       'batch-15', 'batch-16', 'batch-17', 'batch-19', 'batch-20', 'batch-21', 'batch-23', \
#                       'batch-24', 'batch-25', 'batch-26', 'batch-27', 'batch-29', 'batch-30', 'batch-31', \
#                       'batch-32', 'batch-33', 'batch-34', 'batch-36', 'batch-37', 'batch-38', 'batch-40', \
#                       'batch-41']

batches_to_process = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', \
                      '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', \
                      '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', \
                      '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', \
                      '41', '42']

# batches_to_process = ['01', '02', '03']

if bool_windows:
    raw_data_dir_root = 'F:\\Du-Lab\\raw_data\\Susan_Sumner\\CARDIA_03-16-2020\\'

    raw_data_dir = 'F:\\Du-Lab\\raw_data\\Susan_Sumner\\CARDIA_03-16-2020\\Batch' + str(int(batch_index_str)) + '\\'

    results_dir_root = 'D:\\duxiuxia\\cardia\\preprocessing_ver6\\'

    results_dir_batch_root = results_dir_root + 'batch-' + batch_index_str + '\\'
    results_dir_cmix = results_dir_batch_root + 'cmix\\'
    results_dir_msconvert = results_dir_batch_root + 'msconvert\\'
    results_dir_peak_detection = results_dir_batch_root + 'peak_detection\\'
    results_dir_peak_lists = results_dir_peak_detection + 'peaks\\'
    results_dir_peak_filtering = results_dir_batch_root + 'peak_filtering\\'
    results_dir_peak_correction = results_dir_batch_root + 'peak_correction\\'
    results_dir_isotopes = results_dir_batch_root + 'isotopes\\'
    results_dir_analytes = results_dir_batch_root + 'analytes\\'
    results_dir_workflow = results_dir_batch_root + 'workflow\\'
    results_dir_normalization = results_dir_batch_root + 'normalization\\'
    results_dir_comparison = results_dir_batch_root + 'comparison\\'
    results_dir_test = results_dir_batch_root + 'test\\'
    results_dir_between_batch_alignment = results_dir_root + 'between_batch_alignment\\'
    results_dir_examine_batch_data = results_dir_root + 'examine_batch_data\\'
    results_dir_batch_effect_removal = results_dir_root + 'batch_effect_removal\\'

else:
    peak_lists_dir = '/Users/xdu4/Documents/Duxiuxia/my_projects/adap-big/batch-01/results/peaks/'
    results_dir_alignment = '/Users/xdu4/Documents/Duxiuxia/my_projects/adap-big/batch-01/results/alignment/'
    results_dir_normalization = '/Users/xdu4/Documents/Duxiuxia/my_projects/adap-big/batch-01/results/normalization/'
    results_dir_isotopes = '/Users/xdu4/Documents/Duxiuxia/my_projects/adap-big/batch-01/results/isotopes/'
    results_dir_analytes = '/Users/xdu4/Documents/Duxiuxia/my_projects/adap-big/batch-01/results/analytes/'
    results_dir_background = '/Users/xdu4/Documents/Duxiuxia/my_projects/adap-big/batch-01/results/background/'
    results_dir_workflow = '/Users/xdu4/Documents/Duxiuxia/my_projects/adap-big/batch-01/results/workflow/'

# =================================
# mass detection parameters
# =================================
msconvert_exe = 'C:\\Program Files\\ProteoWizard\\ProteoWizard 3.0.19260.3197b7970\\msconvert.exe'

# =================================
# peak detection
# =================================
# adap_big_dir_root = 'C:\\Users\\Du-Lab\\Projects\\CARDIA\\adap-big\\target\\'
adap_big_dir_root = 'C:\\Users\\Du-Lab\\bitbucket\\adap-big\\target\\'

input_jar = adap_big_dir_root + 'input.jar'
chromatogram_builder_jar = adap_big_dir_root + 'chromatogram-builder.jar'
peak_detection_jar = adap_big_dir_root + 'peak-detection.jar'
output_jar = adap_big_dir_root + 'output.jar'

# =================================
# debugging parameters
# =================================
bool_test = False # True, False

bool_check_cmix = False

bool_detect_mass = False

bool_detect_peaks = False

bool_do_within_batch_alignment = False

bool_filter_peaks = False

bool_do_within_batch_normalization = False

bool_do_within_batch_correction = False

bool_between_batch_alignment = True

bool_examine_data_after_between_batch_alignment = False

bool_examine_progenesis_results = False

bool_remove_batch_effect = False

bool_extract_isotopes = False
bool_examine_extracted_isotopes = False

bool_check_workflow = False

bool_extract_analytes = False
bool_group_peaks = False
bool_consolidate_analyte_peaks = False
bool_examine_extracted_analytes = False

bool_do_misc = False

test_num_of_peaks_to_extract_isotopes = 10000
test_num_of_peaks_to_align = 10

# =================================
# plotting_parameters
# =================================
marker_size = 20
marker_color = 'blue'
font_size = 5
font_size = 5
line_width = 1
fig_width = 6
fig_height = 5
dpi = 300
fig_format = '.png'

color_sp = 'magenta'
color_study = 'green'
color_nist = 'blue'
color_blank = 'red'

# =================================
# check cmix parameters
# =================================
cmix_file_name = 'D:\\duxiuxia\\cardia\\doc\\cmix_info.csv'

# =================================
# within-batch alignment  parameters
# =================================
within_batch_alignment_parameters = {}
within_batch_alignment_parameters['bool_use_height'] = False
within_batch_alignment_parameters['method'] = 'join_aligner'
within_batch_alignment_parameters['mz_tolerance_ppm'] = 5
within_batch_alignment_parameters['rt_tolerance'] = 0.2
within_batch_alignment_parameters['mz_weight'] = 0.7
within_batch_alignment_parameters['rt_weight'] = 0.3
within_batch_alignment_parameters['bool_append_unaligned_peaks'] = True

results_dir_alignment = results_dir_batch_root + 'alignment_by_' + within_batch_alignment_parameters['method'] + '\\'


mz_tolerance_ppm = 5

rt_tolerance_small = 0.05
rt_tolerance_medium = 0.1
rt_tolerance_large = 0.2

bool_use_height = False

bool_append_unaligned_peaks = False

rt_tolerance_between_batch_alignment_small = 0.1
rt_tolerance_between_batch_alignment_medium = 0.2
rt_tolerance_between_batch_alignment_large = 0.3

# =================================
# parameters for determining what peaks are background peaks
# =================================
determine_background_parameters = {}
determine_background_parameters['bool_use_aligned_peaks'] = True

determine_background_parameters['method'] = 'yuan'    # 'stats', 'yuan'
determine_background_parameters['snr'] = 3.0

determine_background_parameters['frequency_percentage'] = 0.5
determine_background_parameters['RSD_percentage'] = 30.0
determine_background_parameters['bool_check_peak_intensity'] = False
determine_background_parameters['bool_examine_background_removal'] = False

# =================================
# within-batch normalization
# =================================
normalization_parameters = {}
normalization_parameters['method'] = 'ratio' # 'internal_standard', 'tic', 'none', 'total_peak_intensity', 'ratio'
normalization_parameters['check_normalization_effect'] = False

mz_tryptophan_d5 = 210.129125
rt_tryptophan_d5 = 4.35 # min

# =================================
# filter peaks
# =================================
filter_peaks_parameters = {}
filter_peaks_parameters['sp_to_blank_ratio_threshold'] = 3.0

# =================================
# between batch alignment
# =================================
between_batch_alignment_parameters = {}
between_batch_alignment_parameters['bool_use_raw_intensity'] = True
between_batch_alignment_parameters['bool_use_ppm'] = True
# between_batch_alignment_parameters['mz_tolerance'] = 0.01
# between_batch_alignment_parameters['rt_tolerance'] = 0.1

# =================================
# parameters for extract isotopes
# =================================
isotope_mass_difference = 1.0033
rt_tolerance_isotope = 0.05
correlation_threshold = 0.8
frequency_threshold = 8
correlation_threshold_for_duplicate_peaks = 0.9
percentage_of_highest_peak = 0.75

# =================================
# parameters for extract analytes
# =================================
quantitation_method = 'sum'     # 'frequent', 'rank'

# =================================
# check workflow parameters
# =================================
scaling_method = 'pareto' # 'pareto', 'uv', 'none'
num_of_peaks_for_pca = 10000
num_of_pca_components = 20

# =================================
# parameters for batch effect removal
# =================================
batch_effect_correction_parameters = {}
batch_effect_correction_parameters['bool_between_batch_peak_filtering'] = False
batch_effect_correction_parameters['bool_use_sp'] = False

# =================================
# overall parameters
# =================================
num_of_study_samples_per_repeat_in_one_batch = 10
num_of_repeats_per_batch = 8
num_of_study_samples_per_batch = num_of_repeats_per_batch * num_of_study_samples_per_repeat_in_one_batch
num_of_all_samples_per_batch = num_of_study_samples_per_batch + 3 * num_of_repeats_per_batch

# =================================
# peak filtering by pooled qc parameters
# =================================
detection_rate_threshold = 0.5
rsd_threshold = 0.2
D_ratio_threshold = 0.5

pvalue_threshold = 0.0000005


# =================================
# misc
# =================================
mass_H = 1.007825
