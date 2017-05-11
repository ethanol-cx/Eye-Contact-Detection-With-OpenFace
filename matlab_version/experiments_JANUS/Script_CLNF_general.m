function Script_CLNF_general()

addpath(genpath('../'));

% Replace this with the location of the IJB-FL data location
root_test_data = 'D:/Datasets/janus_labeled';
[images, detections, labels] = Collect_JANUS_imgs(root_test_data);

%% loading the patch experts
   
clmParams = struct;

clmParams.window_size = [25,25; 23,23; 21,21];
clmParams.numPatchIters = size(clmParams.window_size,1);

[patches] = Load_Patch_Experts( '../models/general/', 'ccnf_patches_*_general.mat', [], [], clmParams);

%% Fitting the model to the provided image

% the default PDM to use
pdmLoc = ['../models/pdm/pdm_68_aligned_wild.mat'];

load(pdmLoc);

pdm = struct;
pdm.M = double(M);
pdm.E = double(E);
pdm.V = double(V);

clmParams.regFactor = [35, 27, 20];
clmParams.sigmaMeanShift = [1.25, 1.375, 1.5]; 
clmParams.tikhonov_factor = [2.5, 5, 7.5];

clmParams.startScale = 1;
clmParams.num_RLMS_iter = 10;
clmParams.fTol = 0.01;
clmParams.useMultiScale = true;
clmParams.use_multi_modal = 1;
clmParams.multi_modal_types  = patches(1).multi_modal_types;
   
% Loading the final scale
[clmParams_inner, pdm_inner] = Load_CLM_params_inner();
clmParams_inner.window_size = [17,17;19,19;21,21;23,23];
inds_inner = 18:68;
[patches_inner] = Load_Patch_Experts( '../models/general/', 'ccnf_patches_*general_no_out.mat', [], [], clmParams_inner);
clmParams_inner.multi_modal_types  = patches_inner(1).multi_modal_types;

% for recording purposes
experiment.params = clmParams;

num_points = numel(M)/3;

shapes_all = zeros(size(labels,2),size(labels,3), size(labels,1));
labels_all = zeros(size(labels,2),size(labels,3), size(labels,1));
lhoods = zeros(numel(images),1);

% Use the multi-hypothesis model, as bounding box tells nothing about
% orientation
multi_view = true;
verbose = false; % set to true to visualise the fitting

tic
for i=1:numel(images)

    image = imread(images(i).img);
    image_orig = image;

    if(size(image,3) == 3)
        image = rgb2gray(image);
    end              

    bbox = detections(i,:);                  

    % have a multi-view version
    if(multi_view)

        views = [0,0,0; 0,-30,0; 0,30,0; 0,-55,0; 0,55,0; 0,0,30; 0,0,-30; 0,-90,0; 0,90,0; 0,-70,40; 0,70,-40];
        views = views * pi/180;                                                                                     

        shapes = zeros(num_points, 2, size(views,1));
        ls = zeros(size(views,1),1);

        % Find the best orientation
        for v = 1:size(views,1)
            [shapes(:,:,v),~,~,ls(v)] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams, 'orientation', views(v,:));                                            
        end

        [lhood, v_ind] = max(ls);

        shape = shapes(:,:,v_ind);

    else
        [shape,~,~,lhood] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams);
    end

    % Perform inner face fitting
    shape_inner = shape(inds_inner,:);

    [ a, R, T, ~, l_params, err] = fit_PDM_ortho_proj_to_2D_no_reg(pdm_inner.M, pdm_inner.E, pdm_inner.V, shape_inner);
    if(a > 0.9)
        g_param = [a; Rot2Euler(R)'; T];

        bbox_2 = [min(shape_inner(:,1)), min(shape_inner(:,2)), max(shape_inner(:,1)), max(shape_inner(:,2))];

        [shape_inner] = Fitting_from_bb(image, [], bbox_2, pdm_inner, patches_inner, clmParams_inner, 'gparam', g_param, 'lparam', l_params);

        % Now after detections incorporate the eyes back
        % into the face model

        shape(inds_inner, :) = shape_inner;

        [ ~, ~, ~, ~, ~, ~, shape_fit] = fit_PDM_ortho_proj_to_2D_no_reg(pdm.M, pdm.E, pdm.V, shape);    

        shapes_all(:,:,i) = shape_fit;
    else
        shapes_all(:,:,i) = shape;                    
    end
    labels_all(:,:,i) = labels(i,:,:);

    if(mod(i, 200)==0)
        fprintf('%d done\n', i );
    end

    lhoods(i) = lhood;

    if(verbose)
        v_points = sum(squeeze(labels(i,:,:)),2) > 0;
        DrawFaceOnFig(image_orig, shape, bbox, v_points);
    end

end
toc

experiment.errors_normed = compute_error(labels_all, shapes_all - 1.0);
experiment.lhoods = lhoods;
experiment.shapes = shapes_all;
experiment.labels = labels_all;

fprintf('Done: mean normed error %.3f median normed error %.4f\n', ...
    mean(experiment.errors_normed), median(experiment.errors_normed));

%%
output_results = 'results/results_clnf_general.mat';
save(output_results, 'experiment');
    
end
