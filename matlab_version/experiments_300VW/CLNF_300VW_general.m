clear; 
addpath(genpath('../'));

output_dir = './CLNF_res_general/';

%% select database and load bb initializations
db_root = 'D:\Datasets\300VW_Dataset_2015_12_14\300VW_Dataset_2015_12_14/';
bb_root = '../..//matlab_runners/Feature Point Experiments/300VW_face_dets/';
extra_dir = 'D:\Datasets\300VW_Dataset_2015_12_14\extra';
[ vid_locs, bboxes, gts_all, invalid_frames ] = CollectTestData(db_root, bb_root, extra_dir);

%% loading the patch experts and the PDM
clmParams = struct;

clmParams.window_size = [25,25; 23,23; 21,21; 19,19; 17,17;];
clmParams.numPatchIters = size(clmParams.window_size,1);

[patches] = Load_Patch_Experts( '../models/general/', 'ccnf_patches_*_general.mat', [], [], clmParams);

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
for i=1:numel(vid_locs)

    vid = VideoReader(vid_locs{i});

    bounding_boxes = bboxes{i};
    preds = [];
    n_frames = size(bounding_boxes,1);
    for f=1:n_frames
        input_image = readFrame(vid);
                        
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
            
            views = [0,0,0; 0,-30,0; 0,30,0; 0,-55,0; 0,55,0; 0,0,30; 0,0,-30; 0,-90,0; 0,90,0; 0,-70,40; 0,70,-40];
            views = views * pi/180;                                                                                     

            shapes = zeros(num_points, 2, size(views,1));
            ls = zeros(size(views,1),1);
            lmark_lhoods = zeros(num_points,size(views,1));
            views_used = zeros(num_points,size(views,1));
            g_params = cell(size(views,1),1);
            l_params = cell(size(views,1),1);
            
            % Find the best orientation
            for v = 1:size(views,1)
                [shapes(:,:,v),g_params{v},l_params{v},ls(v),lmark_lhoods(:,v),views_used(v)] = Fitting_from_bb(input_image, [], bb, pdm, patches, clmParams, 'orientation', views(v,:));                                            
            end

            [lhood, v_ind] = max(ls);
            lmark_lhood = lmark_lhoods(:,v_ind);
            g_param = g_params{v_ind};
            l_param = l_params{v_ind};
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
    gt_landmarks = gts_all{i};

    % Remove unreliable frames
    if(~isempty(invalid_frames{i}))
        preds(:,:,int32(invalid_frames{i}))=[];
    end
    
    if(size(gt_landmarks,3) ~= size(preds,3))
        fprintf('something went wrong with vid %d\n', i);
    end
    
    [vid_name,~,~] = fileparts(vid_locs{i});
    [~,vid_name,~] = fileparts(vid_name);
    save([output_dir, '/', vid_name], 'preds', 'gt_landmarks');
end