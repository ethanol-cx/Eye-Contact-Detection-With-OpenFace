clear;

%% Some useful variables for moving the landmarks to common frame and for mirroring
left_to_frontal_map = [17,28; 18,29; 19,30; 20,31;
                       21,34; 22,32; 23,39; 24,38; 25,37; 26,42; 27,41;
                       28,52; 29,51; 30,50; 31,49; 32,60; 33,59; 34,58;
                       35,63; 36,62; 37,61; 38,68; 39,67];
                   
right_to_frontal_map = [17,28; 18,29; 19,30; 20,31;
                       21,34; 22,36; 23,44; 24,45; 25,46; 26,47; 27,48;
                       28,52; 29,53; 30,54; 31,55; 32,56; 33,57; 34,58;
                       35,63; 36,64; 37,65; 38,66; 39,67];

mirror_inds = [1,17;2,16;3,15;4,14;5,13;6,12;7,11;8,10;18,27;19,26;20,25;21,24;22,23;...
              32,36;33,35;37,46;38,45;39,44;40,43;41,48;42,47;49,55;50,54;51,53;60,56;59,57;...
              61,65;62,64;68,66];
          
%%

% Correct the bounding box location in such a way that actual
% initialization of the landmarks has the smallest distance to GT

% the default PDM to use
pdmLoc = ['../models/pdm/pdm_68_aligned_wild.mat'];
load(pdmLoc);
% Set up frontal and profile mean shapes
num_points = numel(M)/3;
m_frontal = reshape(M, num_points, 3)';
width_model_f = max(m_frontal(1,:)) - min(m_frontal(1,:));
height_model_f = max(m_frontal(2,:)) - min(m_frontal(2,:));

rot = Euler2Rot([0,-70*pi/180,0]); 
m_profile = rot * reshape(M, num_points, 3)';
width_model_p = max(m_profile(1,:)) - min(m_profile(1,:));
height_model_p = max(m_profile(2,:)) - min(m_profile(2,:));

% Get image files
load('menpo_train.mat');
img_names = struct;
for i=1:size(bboxes,2)
    img_names(i).name = bboxes(i).name;
end

load('menpo_mtcnn.mat')

inits_frontal = [];
gt_frontal = [];
widths_frontal = [];
heights_frontal = [];
scales_frontal = [];
txs_frontal = [];
tys_frontal = [];

inits_p = [];
gt_p = [];
widths_p = [];
heights_p = [];
scales_p = [];
txs_p = [];
tys_p = [];

data_dir = 'C:\Users\tbaltrus\Documents\menpo_data_orig/';
for i=1:size(bboxes,1)
              
    % Check if the detection is correct first
    tx_gt =  bboxes(i,1);
    ty_gt =  bboxes(i,2);

    tx_det = dets(i,1);
    ty_det = dets(i,2);

    widths_gt = bboxes(i,3) - bboxes(i,1);
    widths_det = dets(i,3) - dets(i,1);

    heights_gt = bboxes(i,4) - bboxes(i,2);
    heights_det = dets(i,4) - dets(i,2);

    bad_det_1 = abs(1 - widths_gt ./ widths_det) > 0.5;
    bad_det_2 = abs(1 - heights_gt ./ heights_det) > 0.5;

    bad_det_3 = abs((tx_gt - tx_det) ./ widths_det) > 0.4;
    bad_det_4 = abs((ty_gt - ty_det) ./ heights_det) > 0.5;

    width = bboxes(i,3) - bboxes(i,1);
    if(width == 0 || bad_det_1 || bad_det_2 || bad_det_3 || bad_det_4)
        continue;
    end
    
    landmarks = gt_labels{i};
    bounding_box = dets(i,:); 

    width = bounding_box(3) - bounding_box(1);
    height = bounding_box(4) - bounding_box(2);

    % For non-frontal need to flip the view to other dir?    
    if(size(landmarks,1) ==39)
        
        landmark_labels = zeros(68,2); 
        
        % Determine the direction
        sum_c = 0;
        for k=1:11
            step = (landmarks(k+1,1) - landmarks(k,1)) * (landmarks(k+1,2) + landmarks(k,2));
            sum_c = sum_c + step;
        end

        % Flip the detection and ground truth if profile is different
        % direction
        if(sum_c < 0)
            width_img = size(imread([data_dir, img_names(i).name]),2);
            
            if(dets(i,3) ~= 0)
                tx = dets(i,1);
                width_f = dets(i,3) - dets(i,1);

                dets(i,1) = width_img - (tx+width_f);
                dets(i,3) = width_img - tx;
            end            
            tx = bboxes(i,1);
            width_f = bboxes(i,3) - bboxes(i,1);
            
            bboxes(i,1) = width_img - (tx+width_f);
            bboxes(i,3) = width_img - tx;
            
            outline = iterate_piece_wise(landmarks(10:-1:1,:), 9);
            brow = iterate_piece_wise(landmarks(16:-1:13,:), 5);

            landmark_labels(9:17,:) = outline;
            landmark_labels(23:27,:) = brow;

            landmark_labels(right_to_frontal_map(:,2),:) = landmarks(right_to_frontal_map(:,1),:);    
            
            % Flip landmarks left to right
            mirror_lbls = landmark_labels;
            mirror_lbls(mirror_lbls ==0) = nan;
            mirror_lbls(:,1) = width -mirror_lbls(:,1);
            tmp1 = mirror_lbls(mirror_inds(:,1),:);
            tmp2 = mirror_lbls(mirror_inds(:,2),:);            
            mirror_lbls(mirror_inds(:,2),:) = tmp1;
            mirror_lbls(mirror_inds(:,1),:) = tmp2;      
            landmark_labels = mirror_lbls;
            landmark_labels(isnan(landmark_labels)) = 0;
            
        else
            outline = iterate_piece_wise(landmarks(1:10,:), 9);
            brow = iterate_piece_wise(landmarks(13:16,:), 5);
            landmark_labels(1:9,:) = outline;
            landmark_labels(18:22,:) = brow;
            landmark_labels(left_to_frontal_map(:,2),:) = landmarks(left_to_frontal_map(:,1),:);
        end
                             
        widths_p = cat(1, widths_p, width);
        heights_p = cat(1, heights_p, height);

        a = ((0.96 * width / width_model_p) + (0.7901 *height/ height_model_p)) / 2;
%         a = ((width / width_model_p) + (height/ height_model_p)) / 2;        
        
        scales_p = cat(1, scales_p, a);

        tx = (bounding_box(3) + bounding_box(1))/2;
        ty = (bounding_box(4) + bounding_box(2))/2;

        % correct it so that the bounding box is just around the minimum
        % and maximum point in the initialised face
        tx = tx - a*(min(m_profile(1,:)) + max(m_profile(1,:)))/2;
        ty = ty - a*(min(m_profile(2,:)) + max(m_profile(2,:)))/2;
        
        tx = tx - 0.1632 * width;
        ty = ty + 0.0944 * height;
        
        txs_p = cat(1, txs_p, tx);
        tys_p = cat(1, tys_p, ty);
        
        % visualisation of the initial state
        global_params = [a, 0, -70*pi/180, 0, tx, ty]';
        local_params = zeros(numel(E), 1);    
        % shape around which the patch experts will be evaluated in the original image
        [shape2D] = GetShapeOrtho(M, V, local_params, global_params);
        
        valid_points = landmark_labels(:,1) ~= 0;
        
        shape2D_img = shape2D(valid_points,1:2);
        inits_p = cat(3, inits_p, shape2D_img);
        gt_p = cat(3, gt_p, landmark_labels(valid_points,:));
%         if(sum_c>0)
%             hold off;imshow(imread([data_dir, img_names(i).name]));hold on;plot(shape2D_img(:,1), shape2D_img(:,2), '.r');hold on;rectangle('Position', [bounding_box(1), bounding_box(2), bounding_box(3)-bounding_box(1), bounding_box(4)-bounding_box(2)]);
%         else
%             hold off;imshow(fliplr(imread([data_dir, img_names(i).name])));hold on;plot(shape2D_img(:,1), shape2D_img(:,2), '.r');hold on;rectangle('Position', [bounding_box(1), bounding_box(2), bounding_box(3)-bounding_box(1), bounding_box(4)-bounding_box(2)]);            
%         end
    else
        widths_frontal = cat(1, widths_frontal, width);
        heights_frontal = cat(1, heights_frontal, height);

        a = ((0.8283 * width / width_model_f) + (0.8454 *height/ height_model_f)) / 2;
%         a = ((width / width_model_f) + (height/ height_model_f)) / 2;        
        
        scales_frontal = cat(1, scales_frontal, a);

        tx = (bounding_box(3) + bounding_box(1))/2;
        ty = (bounding_box(4) + bounding_box(2))/2;

        % correct it so that the bounding box is just around the minimum
        % and maximum point in the initialised face
        tx = tx - a*(min(m_frontal(1,:)) + max(m_frontal(1,:)))/2;
        ty = ty - a*(min(m_frontal(2,:)) + max(m_frontal(2,:)))/2;

        tx = tx - 0.00387 * width;
        ty = ty + 0.1469 * height;
%         
        txs_frontal = cat(1, txs_frontal, tx);
        tys_frontal = cat(1, tys_frontal, ty);
        
        % visualisation of the initial state
        global_params = [a, 0, 0, 0, tx, ty]';
        local_params = zeros(numel(E), 1);    
        % shape around which the patch experts will be evaluated in the original image
        [shape2D] = GetShapeOrtho(M, V, local_params, global_params);
        shape2D_img = shape2D(:,1:2);
        inits_frontal = cat(3, inits_frontal, shape2D_img);
        gt_frontal = cat(3, gt_frontal, landmarks);
%         hold off;imshow(imread([data_dir, img_names(i).name]));hold on;plot(shape2D_img(:,1), shape2D_img(:,2), '.r');hold on;rectangle('Position', [bounding_box(1), bounding_box(2), bounding_box(3)-bounding_box(1), bounding_box(4)-bounding_box(2)]);

    end

end

%% Determine parameters for frontal
widths_frontal = repmat(widths_frontal, 1, 68)';
widths_frontal = widths_frontal(:);
heights_frontal = repmat(heights_frontal, 1, 68)';
heights_frontal = heights_frontal(:);
scales_frontal = repmat(scales_frontal, 1, 68)';
scales_frontal = scales_frontal(:);
txs_frontal = repmat(txs_frontal, 1, 68)';
txs_frontal = txs_frontal(:);
tys_frontal = repmat(tys_frontal, 1, 68)';
tys_frontal = tys_frontal(:);

xs_gt = squeeze(gt_frontal(:,1,:));
xs_gt = xs_gt(:);
xs_init = squeeze(inits_frontal(:,1,:));
xs_init = xs_init(:);

m_frontal_x = m_frontal(1,:);
m_frontal_x = repmat(m_frontal_x, size(gt_frontal,3),1)';
m_frontal_x = m_frontal_x(:);

to_pred = (xs_gt - txs_frontal)./widths_frontal;

feats = [scales_frontal .* m_frontal_x./widths_frontal, widths_frontal./widths_frontal];
frontal_w = feats \ to_pred;
s_width_f = frontal_w(1);
s_tx_f = frontal_w(2);

ys_gt = squeeze(gt_frontal(:,2,:));
ys_gt = ys_gt(:);
ys_init = squeeze(inits_frontal(:,2,:));
ys_init = ys_init(:);

m_frontal_y = m_frontal(2,:);
m_frontal_y = repmat(m_frontal_y, size(gt_frontal,3),1)';
m_frontal_y = m_frontal_y(:);

to_pred = (ys_gt - tys_frontal)./heights_frontal;

feats = [scales_frontal .* m_frontal_y./heights_frontal, heights_frontal./heights_frontal];
frontal_w = feats \ to_pred;
s_height_f= frontal_w(1);
s_ty_f = frontal_w(2);

fprintf('Frontal err x - %.3f, err y - %.3f\n', mean(abs(xs_gt-xs_init)./widths_frontal), mean(abs(ys_gt-ys_init)./heights_frontal));
fprintf('Frontal offs x - %.3f, offs y - %.3f\n', mean((xs_gt-xs_init)./widths_frontal), mean((ys_gt-ys_init)./heights_frontal));

%%
widths_p = repmat(widths_p, 1, 37)';
widths_p = widths_p(:);
heights_p = repmat(heights_p, 1, 37)';
heights_p = heights_p(:);
scales_p = repmat(scales_p, 1, 37)';
scales_p = scales_p(:);
txs_p = repmat(txs_p, 1, 37)';
txs_p = txs_p(:);
tys_p = repmat(tys_p, 1, 37)';
tys_p = tys_p(:);

% Perform correction due to profile
tx_off = (min(m_profile(1,valid_points)) + max(m_profile(1,valid_points)))/2;
txs_p = txs_p + tx_off .* scales_p;

ty_off = (min(m_profile(2,valid_points)) + max(m_profile(2,valid_points)))/2;
tys_p = tys_p + ty_off .* scales_p;

xs_gt = squeeze(gt_p(:,1,:));
xs_gt = xs_gt(:);
xs_init = squeeze(inits_p(:,1,:));
xs_init = xs_init(:);

m_p_x = m_profile(1,valid_points);
m_p_x = repmat(m_p_x, size(gt_p,3),1)';
m_p_x = m_p_x(:);

to_pred = (xs_gt - txs_p)./widths_p;

feats = [scales_p .* m_p_x./widths_p, widths_p./widths_p];
p_w = feats \ to_pred;
s_width_p = p_w(1);
s_tx_p = p_w(2);

ys_gt = squeeze(gt_p(:,2,:));
ys_gt = ys_gt(:);
ys_init = squeeze(inits_p(:,2,:));
ys_init = ys_init(:);

m_p_y = m_profile(2,valid_points);
m_p_y = repmat(m_p_y, size(gt_p,3),1)';
m_p_y = m_p_y(:);

to_pred = (ys_gt - tys_p)./heights_p;

feats = [scales_p .* m_p_y./heights_p, heights_p./heights_p];
p_w = feats \ to_pred;
s_height_p = p_w(1);
s_ty_p = p_w(2);

fprintf('Profile err x - %.3f, err y - %.3f\n', mean(abs(xs_gt-xs_init)./widths_p), mean(abs(ys_gt-ys_init)./heights_p));
fprintf('Profile offs x - %.3f, offs y - %.3f\n', mean((xs_gt-xs_init)./widths_p), mean((ys_gt-ys_init)./heights_p));

% Without any corrections
% Frontal err x - 0.116, err y - 0.150
% Frontal offs x - 0.039, offs y - 0.147
% Profile err x - 0.414, err y - 0.148
% Profile offs x - -0.332, offs y - 0.143

% % With original v1 corrections
% Frontal err x - 0.110, err y - 0.107
% Frontal offs x - 0.047, offs y - -0.104
% Profile err x - 0.413, err y - 0.111
% Profile offs x - -0.313, offs y - -0.106

% With more specific v2 corrections
% Frontal err x - 0.109, err y - 0.105
% Frontal offs x - 0.044, offs y - -0.102
% Profile err x - 0.417, err y - 0.116
% Profile offs x - -0.259, offs y - -0.112

% With V3 corrections
% Frontal err x - 0.109, err y - 0.044
% Frontal offs x - 0.044, offs y - -0.007
% Profile err x - 0.451, err y - 0.065
% Profile offs x - -0.155, offs y - 0.044