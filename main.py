# Author: Xiuxia Du
# March 13, 2020

import os

import config

from check_cmix import checkCmix
from detect_mass import detectMass
from detect_peaks import detectPeaks
from within_batch_alignment import withinBatchAlignment as wba
from filter_peaks import filterPeaks
from within_batch_correction import withinBatchCorrection
from extract_isotopes import extractIsotopes
from extract_analytes import extractAnalytes
from check_workflow import checkWorkflow
from within_batch_normalization import withinBatchNormalization
from between_batch_alignment import betweenBatchAlignment
from examine_data_after_between_batch_alignment import examineDataAfterBetweenBatchAlignment
from remove_batch_effect import removeBatchEffect

from examine_progenesis_results import examineProgenesisResults

# =================================
# main
# =================================
def main():

    # create folder to store results for all batches
    if not os.path.exists(config.results_dir_root):
        os.mkdir(config.results_dir_root)

    print('Current working in ...')
    print(config.results_dir_root)

    # =============================================================
    # batch by batch processing
    # =============================================================
    # create folder to store results for current bach
    print('Batch ' + config.batch_index_str + ' ...')

    if not os.path.exists(config.results_dir_batch_root):
        os.mkdir(config.results_dir_batch_root)

    # -----------------------------
    # check C-mix
    # -----------------------------
    if config.bool_check_cmix:
        print('    Checking cmix ...')
        object_check_cmix = checkCmix()
        object_check_cmix.check_cmix()

    # -----------------------------
    # mass detection
    # -----------------------------
    if config.bool_detect_mass:
        print('    Detecting masses ...')
        object_detect_mass = detectMass()
        object_detect_mass.detect_mass()

    # -----------------------------
    # peak picking
    # -----------------------------
    if config.bool_detect_peaks:
        print('    Detecting peaks ...')
        object_detect_peaks = detectPeaks()
        object_detect_peaks.detect_peaks()

    # -----------------------------
    # within_batch_alignment
    # -----------------------------
    if config.bool_do_within_batch_alignment:
        print('    Within batch alignment ...')
        print('    mz_tolerance = ' + str(config.within_batch_alignment_parameters['mz_tolerance_ppm']))
        print('    rt_tolerance = ' + str(config.within_batch_alignment_parameters['rt_tolerance']))
        print('    mz_weight = ' + str(config.within_batch_alignment_parameters['mz_weight']))
        print('    rt_weight = ' + str(config.within_batch_alignment_parameters['rt_weight']))
        object_within_batch_alignment = wba()
        object_within_batch_alignment.do_within_batch_alignment()

    # -----------------------------
    # filter peaks
    # -----------------------------
    if config.bool_filter_peaks:
        print('    Within batch peak filtering ...')
        object_filter_peak_by_pooled_qc = filterPeaks()
        object_filter_peak_by_pooled_qc.filter_peaks()

    # -----------------------------
    # within-batch normalization
    # -----------------------------
    if config.bool_do_within_batch_normalization:
        print('    Within batch normalization ...')
        object_within_batch_normalization = withinBatchNormalization()
        object_within_batch_normalization.do_within_batch_normalization()

    # -----------------------------
    # within batch correction of peak intensity
    # -----------------------------
    if config.bool_do_within_batch_correction:
        print('    Within batch peak correction ...')
        object_within_batch_correctoin = withinBatchCorrection()
        object_within_batch_correctoin.correct_peak_intensity()

    # -----------------------------
    # between-batch alignment
    # -----------------------------
    if config.bool_between_batch_alignment:
        print('Between batch alignment ...')
        object_between_batch_alignment = betweenBatchAlignment()
        object_between_batch_alignment.do_between_batch_alignment()

    # -----------------------------
    # examine data after between-batch alignment
    # -----------------------------
    if config.bool_examine_data_after_between_batch_alignment:
        print('Examine data after between batch alignment ...')
        object_examine_batch_data = examineDataAfterBetweenBatchAlignment()
        object_examine_batch_data.examine_data()

    # -----------------------------
    # remove batch effect
    # -----------------------------
    if config.bool_remove_batch_effect:
        print('Remove batch effect ...')
        object_remove_batch_effect = removeBatchEffect()
        object_remove_batch_effect.remove_batch_effect()

    # -----------------------------
    # extract isotopes
    # -----------------------------
    if config.bool_extract_isotopes:
        print('Extracting isotopes ...')
        object_extract_isotopes = extractIsotopes()
        object_extract_isotopes.extract_isotopes()

    # -----------------------------
    # extract analytes
    # -----------------------------
    if config.bool_extract_analytes:
        print('Peak grouping and consolidation ...')
        object_extract_analytes = extractAnalytes()
        object_extract_analytes.extract_analytes()

    # -----------------------------
    # check within-batch workflow
    # -----------------------------
    if config.bool_check_workflow:
        print('Within batch PCA ...')
        object_check_workflow = checkWorkflow()
        object_check_workflow.check_workflow()

    # -----------------------------
    # examine progenesis results
    # -----------------------------
    if config.bool_examine_progenesis_results:
        print('examine progenesis results ...')
        object_examine_progenesis_results = examineProgenesisResults()
        object_examine_progenesis_results.examine_progenesis_results()

    print('Batch ' + config.batch_index_str + ' preprocessing done!')

if __name__ == "__main__":
    main()