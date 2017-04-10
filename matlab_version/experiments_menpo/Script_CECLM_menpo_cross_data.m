function Script_CECLM_menpo_cross_data()

addpath('../PDM_helpers/');
addpath('../fitting/normxcorr2_mex_ALL');
addpath('../fitting/');
addpath('../CCNF/');
addpath('../models/');

[images, detections, labels] = Collect_menpo_imgs('D:\Datasets\menpo/');

%% loading the patch experts
   
clmParams = struct;

clmParams.window_size = [25,25; 23,23; 21,21; 21,21];
clmParams.numPatchIters = size(clmParams.window_size,1);

[patches] = Load_DCLM_Patch_Experts( '../models/cen/', 'cen_patches_*_general.mat', [], [], clmParams);

%% Fitting the model to the provided image

output_root = './menpo_fit_dclm_more_epochs/';

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
clmParams.numPatchIters = 4;

% for recording purposes
experiment.params = clmParams;

num_points = numel(M)/3;

shapes_all = cell(numel(images), 1);
labels_all = cell(numel(images), 1);
lhoods = zeros(numel(images),1);
all_lmark_lhoods = zeros(num_points, numel(images));
all_views_used = zeros(numel(images),1);

% Use the multi-hypothesis model, as bounding box tells nothing about
% orientation
multi_view = true;
verbose = false;
tic
if(verbose)
    f = figure;
end

for i=1:numel(images)

    image = imread(images(i).img);
    image_orig = image;
    
    if(size(image,3) == 3)
        image = rgb2gray(image);
    end              

    bbox = squeeze(detections(i,:));                  
    
    % have a multi-view version
    if(multi_view)

        views = [0,0,0; 0,-70,40; 0,70,-40; 0,-30,0; 0,-60,0; 0,-90,0; 0,30,0; 0,60,0; 0,90,0; 0,0,30; 0,0,-30;];
        views = views * pi/180;                                                                                     

        shapes = zeros(num_points, 2, size(views,1));
        ls = zeros(size(views,1),1);
        lmark_lhoods = zeros(num_points,size(views,1));
        views_used = zeros(size(views,1),1);

        % Find the best orientation
        for v = 1:size(views,1)
            [shapes(:,:,v),~,~,ls(v),lmark_lhoods(:,v),views_used(v)] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams, 'orientation', views(v,:));                                            
        end

        [lhood, v_ind] = max(ls);
        lmark_lhood = lmark_lhoods(:,v_ind);

        shape = shapes(:,:,v_ind);
        view_used = views_used(v_ind);

    else
        [shape,~,~,lhood,lmark_lhood,view_used] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams);
    end

    all_lmark_lhoods(:,i) = lmark_lhood;
    all_views_used(i) = view_used;

    shapes_all{i} = shape;
    labels_all{i} = labels{i};

    if(mod(i, 200)==0)
        fprintf('%d done\n', i );
    end

    lhoods(i) = lhood;

    if(verbose)

        actualShape = labels{i};
        %         f = figure('visible','off');

        try
        if(max(image_orig(:)) > 1)
            imshow(double(image_orig)/255, 'Border', 'tight');
        else
            imshow(double(image_orig), 'Border', 'tight');
        end
        axis equal;
        hold on;
        v_points = logical(patches(1).visibilities(view_used,:))';
        plot(shape(v_points,1), shape(v_points,2),'.r','MarkerSize',20);
        plot(shape(v_points,1), shape(v_points,2),'.b','MarkerSize',10);
%                                         print(f, '-r80', '-dpng', sprintf('%s/%s%d.png', output_root, 'fit', i));
        print(f, '-djpeg', sprintf('%s/%s%d.jpg', output_root, 'fit', i));
%                                         close(f);
        hold off;
        drawnow expose
        catch warn

        end
    end

end
toc

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

%%
output_results = 'results/results_ceclm_cross-data.mat';
save(output_results, 'experiments');
    
end
