function Script_CECLM_general()

addpath('../PDM_helpers/');
addpath('../fitting/normxcorr2_mex_ALL');
addpath('../fitting/');
addpath('../CCNF/');
addpath('../models/');

% Replace this with the location of in 300 faces in the wild data
root_test_data = 'D:/Datasets/janus_labeled';

[images, detections, labels] = Collect_JANUS_imgs(root_test_data);

%% loading the patch experts
   
clmParams = struct;

clmParams.window_size = [25,25; 23,23; 21,21; 21,21];
clmParams.numPatchIters = size(clmParams.window_size,1);

[patches] = Load_DCLM_Patch_Experts( '../models/cen/', 'cen_patches_*_general.mat', [], [], clmParams);

%% Fitting the model to the provided image

output_root = './wild_fit_dclm/';

% the default PDM to use
pdmLoc = ['../models/pdm/pdm_68_aligned_wild.mat'];

load(pdmLoc);

pdm = struct;
pdm.M = double(M);
pdm.E = double(E);
pdm.V = double(V);

clmParams.regFactor = 0.9*[35, 27, 20, 20];
clmParams.sigmaMeanShift = 1.5*[1.25, 1.375, 1.5, 1.5]; 
clmParams.tikhonov_factor = [2.5, 5, 7.5, 7.5];

clmParams.startScale = 1;
clmParams.num_RLMS_iter = 10;
clmParams.fTol = 0.01;
clmParams.useMultiScale = true;
clmParams.use_multi_modal = 1;
clmParams.multi_modal_types  = patches(1).multi_modal_types;
clmParams.numPatchIters = 4;

% for recording purposes
experiment.params = clmParams;

num_points = numel(M)/3;

shapes_all = zeros(size(labels,2),size(labels,3), size(labels,1));
labels_all = zeros(size(labels,2),size(labels,3), size(labels,1));
lhoods = zeros(numel(images),1);
all_lmark_lhoods = zeros(num_points, numel(images));
all_views_used = zeros(numel(images),1);

% Use the multi-hypothesis model, as bounding box tells nothing about
% orientation
multi_view = true;
verbose = false;

% As the orientations are not equally reliable reweigh them
load('../learn_error_mapping/cen_general_mapping.mat');

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
        
        [shape,~,~,lhood,lmark_lhood,view_used] =...
            Fitting_from_bb_multi_hyp(image, [], bbox, pdm, patches, clmParams, views, early_term_params);
    else
        [shape,~,~,lhood,lmark_lhood,view_used] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams);
    end

    all_lmark_lhoods(:,i) = lmark_lhood;
    all_views_used(i) = view_used;

    shapes_all(:,:,i) = shape;
    labels_all(:,:,i) = labels(i,:,:);

    if(mod(i, 200)==0)
        fprintf('%d done\n', i );
    end

    lhoods(i) = lhood;

    if(verbose)

        actualShape = squeeze(labels(i,:,:));
        
        [height_img, width_img,~] = size(image_orig);
        width = max(actualShape(:,1)) - min(actualShape(:,1));
        height = max(actualShape(:,2)) - min(actualShape(:,2));

        v_points = sum(squeeze(labels(i,:,:)),2) > 0;

        img_min_x = max(int32(min(actualShape(v_points,1))) - width/3,1);
        img_max_x = min(int32(max(actualShape(v_points,1))) + width/3,width_img);

        img_min_y = max(int32(min(actualShape(v_points,2))) - height/3,1);
        img_max_y = min(int32(max(actualShape(v_points,2))) + height/3,height_img);

        shape(:,1) = shape(:,1) - double(img_min_x);
        shape(:,2) = shape(:,2) - double(img_min_y);

        image_orig = image_orig(img_min_y:img_max_y, img_min_x:img_max_x, :);    

        % valid points to draw (not to draw
        % occluded ones)

%         f = figure('visible','off');
%         f = figure;
        try
        if(max(image_orig(:)) > 1)
            imshow(double(image_orig)/255, 'Border', 'tight');
        else
            imshow(double(image_orig), 'Border', 'tight');
        end
        axis equal;
        hold on;
        
        plot(shape(v_points,1), shape(v_points,2),'.r','MarkerSize',20);
        plot(shape(v_points,1), shape(v_points,2),'.b','MarkerSize',10);
%                                         print(f, '-r80', '-dpng', sprintf('%s/%s%d.png', output_root, 'fit', i));
%         print(f, '-djpeg', sprintf('%s/%s%d.jpg', output_root, 'fit', i));
%                                         close(f);
        hold off;
        drawnow expose
%         close(f);
        catch warn

        end
    end

end
toc

experiment.errors_normed = compute_error(labels_all, shapes_all + 0.5);
experiment.lhoods = lhoods;
experiment.shapes = shapes_all;
experiment.labels = labels_all;
experiment.all_lmark_lhoods = all_lmark_lhoods;
experiment.all_views_used = all_views_used;
% save the experiment
if(~exist('experiments', 'var'))
    experiments = experiment;
else
    experiments = cat(1, experiments, experiment);
end
fprintf('experiment %d done: mean normed error %.3f median normed error %.4f\n', ...
    numel(experiments), mean(experiment.errors_normed), median(experiment.errors_normed));

%%
output_results = 'results/results_ceclm_general.mat';
save(output_results, 'experiments');
    
end
