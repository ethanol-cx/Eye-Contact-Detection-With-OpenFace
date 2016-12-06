clear; 
addpath('../PDM_helpers/');
addpath('../fitting/normxcorr2_mex_ALL');
addpath('../fitting/');
addpath('../CCNF/');
addpath('../models/');

output_dir = './DCLM_res/';

%% select database and load bb initializations
db_root = 'D:\Datasets\300VW_Dataset_2015_12_14\300VW_Dataset_2015_12_14/';
bb_root = '../..//matlab_runners/Feature Point Experiments/300VW_face_dets/';
extra_dir = 'D:\Datasets\300VW_Dataset_2015_12_14\extra';

in_dirs = dir(db_root);
in_dirs = in_dirs(3:end);
in_dirs = in_dirs(1:end-2);
preds_all = [];
gts_all = [];

%% loading the patch experts and the PDM
clmParams = struct;

clmParams.window_size = [25,25; 23,23; 21,21; 21,21];
clmParams.numPatchIters = size(clmParams.window_size,1);

[patches] = Load_DCLM_Patch_Experts( '../models/general/', 'dccnf_patches_*_general.mat', [], [], clmParams);

% the default PDM to use
pdmLoc = ['../models/pdm/pdm_68_aligned_wild.mat'];

load(pdmLoc);

pdm = struct; pdm.M = double(M); pdm.E = double(E); pdm.V = double(V);
num_points = numel(pdm.M) / 3;

clmParams.regFactor = [35, 27, 20, 20];
clmParams.sigmaMeanShift = [1.25, 1.375, 1.5, 1.5]; 
clmParams.tikhonov_factor = [0,0,0,0];

clmParams.startScale = 1;
clmParams.num_RLMS_iter = 10;
clmParams.fTol = 0.01;
clmParams.useMultiScale = true;
clmParams.use_multi_modal = 1;
clmParams.multi_modal_types  = patches(1).multi_modal_types;
clmParams.numPatchIters = 4;

multi_view = true;
verbose = true;

%% Select video
for i=1:numel(in_dirs)

    in_file_name = [db_root '/', in_dirs(i).name, '/vid.avi']; 

    vid = VideoReader(in_file_name);

    bounding_boxes = dlmread([bb_root,  in_dirs(i).name, '_dets.txt'], ',');
    preds = [];
    n_frames = size(bounding_boxes,1);
    for f=1:n_frames
        input_image = readFrame(vid);
                
        reset = true;
        
        %% Initialize from detected bounding boxes every 30 frames
        if (mod(f-1, 30) == 0)
            ind = min(f, size(bounding_boxes,1));
            bb = bounding_boxes(ind, :);
            % If no face detected use the closest detected BB
            if(bb(3) == 0)
               ind_next = ind + find(bounding_boxes(ind+1:end,3), 1);  
               if(isempty(bounding_boxes(ind_next)) || bounding_boxes(ind_next,3)==0)
                   ind_next = find(bounding_boxes(1:ind,3), 1, 'last');
               end
               bb = bounding_boxes(ind_next, :);
            end
            bb(3) = bb(1) + bb(3);
            bb(4) = bb(2) + bb(4);
            reset = true;
        else
            reset = false;
        end
    
        % have a multi-view version for initialization, otherwise use
        % previous shape
        if(reset && multi_view)
            clmParams.window_size = [25,25; 23,23; 21,21; 21,21];
            clmParams.numPatchIters = 4;
            clmParams.startScale = 1;
            views = [0,0,0; 0,-45,0; -30,0,0; 0,45,0; 30,0,0];
            views = views * pi/180;                                                                                     

            shapes = zeros(num_points, 2, size(views,1));
            ls = zeros(size(views,1),1);
            lmark_lhoods = zeros(num_points,size(views,1));
            views_used = zeros(num_points,size(views,1));

            % Find the best orientation
            for v = 1:size(views,1)
                [shapes(:,:,v),g_param,l_param,ls(v),lmark_lhoods(:,v),views_used(v)] = Fitting_from_bb(input_image, [], bb, pdm, patches, clmParams, 'orientation', views(v,:));                                            
            end

            [lhood, v_ind] = max(ls);
            lmark_lhood = lmark_lhoods(:,v_ind);

            shape = shapes(:,:,v_ind);
            view_used = views_used(v);

        else            
            clmParams.window_size = [23,23; 21,21; 19,19; 17,17];
            clmParams.numPatchIters = 3;
            clmParams.startScale = 2;
            [shape,g_param,l_param,lhood,lmark_lhood,view_used] = Fitting_from_bb(input_image, [], bb, pdm, patches, clmParams, 'gparam', g_param, 'lparam', l_param);
        end        
        
        preds = cat(3, preds, shape);
            
        %% plot the result
        imshow(input_image);
        hold on;
        plot(shape(:,1), shape(:,2), '.r');
%         rectangle('Position', [bb(2), bb(1), bb(4), bb(3)]);
        hold off;
        drawnow expose
        
    end
    
    %% Grab the ground truth
    fps_all = dir([db_root, '/', in_dirs(i).name, '/annot/*.pts']);
    gt_landmarks = zeros([68, 2, size(fps_all)]);
    for k = 1:size(fps_all)

        gt_landmarks_frame = dlmread([db_root, '/', in_dirs(i).name, '/annot/', fps_all(k).name], ' ', 'A4..B71');
        gt_landmarks(:,:,k) = gt_landmarks_frame;
    end
    
    if(size(gt_landmarks,3) ~= size(preds,3))
        fprintf('something went wrong with vid %d\n', i);
    end

    % Remove unreliable frames
    if(exist([extra_dir, '/', in_dirs(i).name, '.mat'], 'file'))
        load([extra_dir, '/', in_dirs(i).name, '.mat']);
        gt_landmarks(:,:,int32(error)) = [];
        preds(:,:,int32(error))=[];
    end    
    
    vid_name = in_dirs(i).name;
    save([output_dir, '/', vid_name], 'preds', 'gt_landmarks');
end
% [pocr_error, err_pp_pocr] = compute_error( gts_all,  preds_all);



