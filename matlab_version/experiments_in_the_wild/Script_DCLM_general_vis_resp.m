function Script_DCLM_general_vis_resp()

addpath('../PDM_helpers/');
addpath('../fitting/normxcorr2_mex_ALL');
addpath('../fitting/');
addpath('../CCNF/');
addpath('../models/');

% Replace this with the location of in 300 faces in the wild data
if(exist([getenv('USERPROFILE') '/Dropbox/AAM/test data/'], 'file'))
    root_test_data = [getenv('USERPROFILE') '/Dropbox/AAM/test data/'];    
else
    root_test_data = 'D:/Dropbox/Dropbox/AAM/test data/';
end

[images, detections, labels] = Collect_wild_imgs(root_test_data);

to_vis = [29,51,75,79,81,9,112,146,152,172,199,204,230,234,235,241,251,256,263,272,279,340,342,348,358,362,394,478,484,486];
images = images(to_vis);
detections = detections(to_vis,:);
labels = labels(to_vis,:);

%% loading the patch experts
   
clmParams = struct;

clmParams.window_size = [31,31;];
clmParams.numPatchIters = size(clmParams.window_size,1);

[patches] = Load_DCLM_Patch_Experts( '../models/general/', 'dccnf_patches_*_general.mat', [], [], clmParams);

%% Fitting the model to the provided image

out_dir_root = './patch_expert_responses/';

if(~exist(out_dir_root, 'dir'))
    mkdir(out_dir_root);
end

% the default PDM to use
pdmLoc = ['../models/pdm/pdm_68_aligned_wild.mat'];

load(pdmLoc);

pdm = struct;
pdm.M = double(M);
pdm.E = double(E);
pdm.V = double(V);

clmParams.startScale = 1;
clmParams.num_RLMS_iter = 10;
clmParams.fTol = 0.01;
clmParams.useMultiScale = true;
clmParams.use_multi_modal = 1;
clmParams.multi_modal_types  = patches(1).multi_modal_types;
clmParams.numPatchIters = 1;

% for recording purposes
experiment.params = clmParams;

num_points = numel(M)/3;

for i=1:numel(images)

    image = imread(images(i).img);
    image_orig = image;
    
    if(size(image,3) == 3)
        image = rgb2gray(image);
    end              

    bbox = detections(i,:);                  
    out_dir = [out_dir_root, '/', num2str(to_vis(i))];
    if(~exist(out_dir, 'dir'))
        mkdir(out_dir);
    end
    
    Fitting_from_bb_vis(image, [], bbox, pdm, patches, clmParams, out_dir);
  
end
    
end
