function Script_DCLM_general_vis_select()

addpath('../PDM_helpers/');
addpath('../fitting/normxcorr2_mex_ALL');
addpath('../fitting/');
addpath('../CCNF/');
addpath('../models/');

% Replace this with the location of in 300 faces in the wild data
if(exist([getenv('USERPROFILE') '/Dropbox/AAM/test data/'], 'file'))
    root_test_data = [getenv('USERPROFILE') '/Dropbox/AAM/test data/'];    
else
    root_test_data = 'C:\Users\tbaltrus\Documents\DATA/300-W/';
end

[images, detections, labels] = Collect_wild_imgs_train(root_test_data);

% Only use a subset of images as otherwise it will take a bit too long
rng(0);
subset = randperm(numel(images), 750);
images = images(subset);
detections = detections(subset,:);
labels = labels(subset,:,:);

%% loading the patch experts
   
clmParams = struct;

clmParams.window_size = [25,25; 23,23; 21,21;];
clmParams.numPatchIters = size(clmParams.window_size,1);

[patches] = Load_DCLM_Patch_Experts( '../models/dpn/', 'dpn_patches_*_general.mat', [], [], clmParams);

%% Fitting the model to the provided image

output_root = './wild_fit_dclm/';

% the default PDM to use
pdmLoc = ['../models/pdm/pdm_68_aligned_wild.mat'];

load(pdmLoc);

pdm = struct;
pdm.M = double(M);
pdm.E = double(E);
pdm.V = double(V);

clmParams.regFactor = [35, 27, 20, 20];
clmParams.sigmaMeanShift = [1.25, 1.375, 1.5, 1.5]; 
clmParams.tikhonov_factor = [2.5, 5, 7.5, 7.5];

clmParams.startScale = 1;
clmParams.num_RLMS_iter = 10;
clmParams.fTol = 0.01;
clmParams.useMultiScale = true;
clmParams.use_multi_modal = 1;
clmParams.multi_modal_types  = patches(1).multi_modal_types;
clmParams.numPatchIters = 3;

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
tic

experiments_full = struct;

best_so_far = [66;62;54;60;38;5;30;13;28;59;44];

% The first patch removal
to_rem_1 = [4;68;58;62;51;6;59;20;63;53;25;56;14;64;9;67;2;33;11;37;17;52;26;60;28;34;44;38;29;8;21;15;12;18];
to_rem_2 = [6;62;50;25;59;20;17;66;64;57;39;14;12;68;41;45;34;43;30;60;4;29;1;61;47;9;65;52;37;22;15;35;54;58];
to_rem_from = [1,2,3,6,7];
% patches(1).visibilities(to_rem_from,to_rem_init) = 0;

% A greedy method for removing visibilities
for to_rem = 1:23

    % Not ideal ones to turn off, but a much faster version
    all_ids = 1:68;
    visibility_off = setdiff(all_ids, best_so_far);
    inds = randperm(numel(visibility_off), 20);
    visibility_off = visibility_off(inds);
    
    % for recording purposes
    experiment.params = clmParams;

    for w=visibility_off

        patches(1).visibilities(to_rem_from,to_rem_1) = 0;
        patches(2).visibilities(to_rem_from,to_rem_2) = 0;
        patches(3).visibilities(to_rem_from,:) = 1;
        to_rem_c = cat(1, best_so_far, w);
        if(w > 0)
            patches(3).visibilities(to_rem_from,to_rem_c) = 0;
        end
        for i=1:numel(images)

            image = imread(images(i).img);
            image_orig = image;

            if(size(image,3) == 3)
                image = rgb2gray(image);
            end              

            bbox = detections(i,:);                  

            % have a multi-view version
            if(multi_view)

                views = [0,0,0; 0,-30,0; -30,0,0; 0,30,0; 30,0,0];
                views = views * pi/180;                                                                                     

                shapes = zeros(num_points, 2, size(views,1));
                ls = zeros(size(views,1),1);
                lmark_lhoods = zeros(num_points,size(views,1));
                views_used = zeros(num_points,size(views,1));

                % Find the best orientation
                for v = 1:size(views,1)
                    [shapes(:,:,v),~,~,ls(v),lmark_lhoods(:,v),views_used(v)] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams, 'orientation', views(v,:));                                            
                end

                [lhood, v_ind] = max(ls);
                lmark_lhood = lmark_lhoods(:,v_ind);

                shape = shapes(:,:,v_ind);
                view_used = views_used(v);

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

                img_min_x = max(int32(min(actualShape(:,1))) - width/3,1);
                img_max_x = min(int32(max(actualShape(:,1))) + width/3,width_img);

                img_min_y = max(int32(min(actualShape(:,2))) - height/3,1);
                img_max_y = min(int32(max(actualShape(:,2))) + height/3,height_img);

                shape(:,1) = shape(:,1) - double(img_min_x);
                shape(:,2) = shape(:,2) - double(img_min_y);

                image_orig = image_orig(img_min_y:img_max_y, img_min_x:img_max_x, :);    

                % valid points to draw (not to draw
                % occluded ones)
                v_points = sum(squeeze(labels(i,:,:)),2) > 0;

        %         f = figure('visible','off');
                f = figure;
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
                close(f);
                catch warn

                end
            end

        end
        toc

        experiment.errors_normed = compute_error(labels_all - 0.5, shapes_all);
        experiment.v_off = w;
        experiment.err = mean(experiment.errors_normed);
        experiment.err_med = median(experiment.errors_normed);
        % save the experiment
        if(~exist('experiments', 'var'))
            experiments = experiment;
        else
            experiments = cat(1, experiments, experiment);
        end
        fprintf('experiment %d done: mean normed error %.3f median normed error %.4f\n', ...
            numel(experiments), mean(experiment.errors_normed), median(experiment.errors_normed));

    end
    
    % Grab the best suceeding switch off
    [~,id] = min(cat(1, experiments.err));
    best_so_far = cat(1, best_so_far, experiments(id).v_off);
    experiments_full(numel(best_so_far)).best_so_far = best_so_far;
    experiments_full(numel(best_so_far)).error_mean = experiments(id).err;
    experiments_full(numel(best_so_far)).error_med = experiments(id).err_med;
    output_results = 'results/results_wild_dclm_vis_off_scale3.mat';
    save(output_results, 'experiments_full'); 
    clear experiments
    clear experiment
end
%%

    
end
