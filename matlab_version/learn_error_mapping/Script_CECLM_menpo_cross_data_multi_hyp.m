function Script_CECLM_menpo_cross_data_multi_hyp()

addpath('../PDM_helpers/');
addpath('../fitting/normxcorr2_mex_ALL');
addpath('../fitting/');
addpath('../CCNF/');
addpath('../models/');

[images, detections, labels] = Collect_menpo_train_imgs('D:\Datasets\menpo/');

%% loading the patch experts
   
clmParams = struct;

clmParams.window_size = [25,25; 23,23; 21,21; 21,21];
clmParams.numPatchIters = size(clmParams.window_size,1);

[patches] = Load_DCLM_Patch_Experts( '../models/cen/', 'cen_patches_*_menpo.mat', [], [], clmParams);

%% Fitting the model to the provided image

% the default PDM to use
pdmLoc = ['../models/pdm/pdm_68_aligned_wild.mat'];

load(pdmLoc);

pdm = struct;
pdm.M = double(M);
pdm.E = double(E);
pdm.V = double(V);

clmParams.regFactor = 0.9 * [35, 27, 20, 20];
clmParams.sigmaMeanShift = 1.5 * [1.25, 1.375, 1.5, 1.5]; 
clmParams.tikhonov_factor = [2.5, 5, 7.5, 7.5];

clmParams.startScale = 1;
clmParams.num_RLMS_iter = 10;
clmParams.fTol = 0.01;
clmParams.useMultiScale = true;
clmParams.use_multi_modal = 1;
clmParams.multi_modal_types  = patches(1).multi_modal_types;

views = [0,0,0; 0,-30,0; 0,30,0; 0,-55,0; 0,55,0; 0,0,30; 0,0,-30; 0,-90,0; 0,90,0; 0,-70,40; 0,70,-40];
views = views * pi/180;                                                                                     
num_views = size(views,1);

clmParams.numPatchIters = 1;

% for recording purposes
experiment.params = clmParams;

num_points = numel(M)/3;

shapes_all = cell(numel(images), num_views);
labels_all = cell(numel(images), 1);
lhoods = zeros(numel(images),num_views);
all_lmark_lhoods = zeros(num_points, numel(images),num_views);
all_views_used = zeros(numel(images),num_views);
errors_view = zeros(numel(images),num_views);
% Use the multi-hypothesis model, as bounding box tells nothing about
% orientation
verbose = true;
tic

for i=1:numel(images)

    image = imread(images(i).img);
    image_orig = image;

    if(size(image,3) == 3)
        image = rgb2gray(image);
    end              

    bbox = squeeze(detections(i,:));                  


    % Find the best orientation
    for v = 1:size(views,1)
        [shape,~,~,lhood,lmark_lhoods,view_used] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams, 'orientation', views(v,:));                                            
        shapes_all{i,v} = shape;
        all_views_used(i,v) = view_used;
        all_lmark_lhoods(:, i,v) = lmark_lhoods;
        lhoods(i,v) = lhood;
        errors_view(i,v) = compute_error_menpo_1(labels(i), {shape});
    end

    labels_all{i} = labels{i};

    if(mod(i, 200)==0)
        fprintf('%d done\n', i );
    end

end
toc

experiment.lhoods = lhoods;
experiment.errors_view = errors_view;
experiment.all_views_used = all_views_used;

%%
output_results = ['results/results_ceclm_menpo.mat'];
save(output_results, 'experiment');
end
