% This is sort of the unit test for the whole module (needs datasets)
% Will take several hours to run all
clear
tic
%% Head pose
cd('Head Pose Experiments');
run_head_pose_tests_OpenFace_CECLM;
assert(median(all_errors_biwi_OF(:)) < 3.0);
assert(median(all_errors_bu_OF(:)) < 1.9);
assert(median(all_errors_ict_OF(:)) < 2.1);
run_head_pose_tests_OpenFace;
assert(median(all_errors_biwi_OF(:)) < 2.7);
assert(median(all_errors_bu_OF(:)) < 2.0);
assert(median(all_errors_ict_OF(:)) < 2.0);
cd('../');

%% Features
cd('Feature Point Experiments');
run_OpenFace_feature_point_tests_300W;
assert(median(err_ceclm) < 0.036);
assert(median(err_clnf) < 0.039);
run_yt_dataset;
assert(median(ceclm_error) < 0.045);
assert(median(clnf_error) < 0.053);
run_300VW_dataset_OpenFace;
assert(median(ceclm_error_49_cat_1) < 0.025);
assert(median(ceclm_error_49_cat_2) < 0.027);
assert(median(ceclm_error_49_cat_3) < 0.032);
assert(median(ceclm_error_66_cat_1) < 0.032);
assert(median(ceclm_error_66_cat_2) < 0.036);
assert(median(ceclm_error_66_cat_3) < 0.041);

assert(median(clnf_error_49_cat_1) < 0.029);
assert(median(clnf_error_49_cat_2) < 0.035);
assert(median(clnf_error_49_cat_3) < 0.040);
assert(median(clnf_error_66_cat_1) < 0.039);
assert(median(clnf_error_66_cat_2) < 0.044);
assert(median(clnf_error_66_cat_3) < 0.049);

cd('../');

%% AUs
cd('Action Unit Experiments');
run_AU_prediction_DISFA
assert(mean(au_res) > 0.7);

run_AU_prediction_SEMAINE
assert(mean(f1s) > 0.40);

run_AU_prediction_FERA2011
assert(mean(au_res) > 0.53);

cd('../');

%% Gaze
cd('Gaze Experiments');
extract_mpii_gaze_test
assert(mean_error < 8.8)
assert(median_error < 8.1)
cd('../');

%% Demos
cd('Demos');
run_demo_images;
run_demo_videos;
run_demo_video_multi;
feature_extraction_demo_vid;
feature_extraction_demo_img_seq;
gaze_extraction_demo_vid;
cd('../');
toc