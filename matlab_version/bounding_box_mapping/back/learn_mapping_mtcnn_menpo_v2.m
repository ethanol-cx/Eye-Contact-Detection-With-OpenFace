clear;
% Get image files
load('menpo_train.mat');
img_names = struct;
for i=1:size(bboxes,2)
    img_names(i).name = bboxes(i).name;
end

load('menpo_mtcnn.mat')

frontal = true(size(bboxes,1), 1);
right = false(size(bboxes,1), 1);

data_dir = 'C:\Users\tbaltrus\Documents\menpo_data_orig/';
for i=1:size(bboxes,1)
              
    % For non-frontal need to flip the view to other dir?    
    if(size(gt_labels{i},1) ==39)
        frontal(i) = false;
        landmarks = gt_labels{i};
        % Determine the direction
        sum_c = 0;
        for k=1:11
            step = (landmarks(k+1,1) - landmarks(k,1)) * (landmarks(k+1,2) + landmarks(k,2));
            sum_c = sum_c + step;
        end
        right(i) = true;

        % Flip the detection and ground truth if profile is different
        % direction
        if(sum_c < 0)
            width = size(imread([data_dir, img_names(i).name]),2);
            
            if(dets(i,3) ~= 0)
                tx = dets(i,1);
                width_f = dets(i,3) - dets(i,1);

                dets(i,1) = width - (tx+width_f);
                dets(i,3) = width - tx;
            end            
            tx = bboxes(i,1);
            width_f = bboxes(i,3) - bboxes(i,1);
            
            bboxes(i,1) = width - (tx+width_f);
            bboxes(i,3) = width - tx;
        end
        
    end

end
bboxes_det = dets;
bboxes_gt = bboxes;

non_detected = bboxes_det(:,3) == 0;

% Removing the outliers
widths_gt = bboxes_gt(:,3) - bboxes_gt(:,1);
widths_det = bboxes_det(:,3) - bboxes_det(:,1);

heights_gt = bboxes_gt(:,4) - bboxes_gt(:,2);
heights_det = bboxes_det(:,4) - bboxes_det(:,2);

tx_gt =  bboxes_gt(:,1);
ty_gt =  bboxes_gt(:,2);

tx_det = bboxes_det(:,1);
ty_det = bboxes_det(:,2);

bad_det_1 = abs(1 - widths_gt ./ widths_det) > 0.5;
bad_det_2 = abs(1 - heights_gt ./ heights_det) > 0.5;

bad_det_3 = abs((tx_gt - tx_det) ./ widths_det) > 0.4;
bad_det_4 = abs((ty_gt - ty_det) ./ heights_det) > 0.5;

non_detected = non_detected | bad_det_1 | bad_det_2 | bad_det_3 | bad_det_4;

%% Perform correction for frontal images
bboxes_gt = bboxes(frontal & (~non_detected),:);
bboxes_det = dets(frontal & (~non_detected),:);

% Find the width and height mappings
widths_gt = bboxes_gt(:,3) - bboxes_gt(:,1);
widths_det = bboxes_det(:,3) - bboxes_det(:,1);

heights_gt = bboxes_gt(:,4) - bboxes_gt(:,2);
heights_det = bboxes_det(:,4) - bboxes_det(:,2);

s_width_f = widths_det \ widths_gt;
s_height_f = heights_det \ heights_gt;

tx_gt =  bboxes_gt(:,1);
ty_gt =  bboxes_gt(:,2);

tx_det = bboxes_det(:,1);
ty_det = bboxes_det(:,2);

s_tx_f = median((tx_gt - tx_det) ./ widths_det);
s_ty_f = median((ty_gt - ty_det) ./ heights_det);

%% Now do the profile views
bboxes_gt = bboxes(right & (~non_detected),:);
bboxes_det = dets(right & (~non_detected),:);

% Find the width and height mappings
widths_gt = bboxes_gt(:,3) - bboxes_gt(:,1);
widths_det = bboxes_det(:,3) - bboxes_det(:,1);

heights_gt = bboxes_gt(:,4) - bboxes_gt(:,2);
heights_det = bboxes_det(:,4) - bboxes_det(:,2);

s_width_p = widths_det \ widths_gt;
s_height_p = heights_det \ heights_gt;

tx_gt =  bboxes_gt(:,1);
ty_gt =  bboxes_gt(:,2);

tx_det = bboxes_det(:,1);
ty_det = bboxes_det(:,2);

s_tx_p = median((tx_gt - tx_det) ./ widths_det);
s_ty_p = median((ty_gt - ty_det) ./ heights_det);

%% Perform the frontal correction
widths_det = dets(frontal & (~non_detected),3) - dets(frontal & (~non_detected),1);
heights_det = dets(frontal & (~non_detected),4) - dets(frontal & (~non_detected),2);
tx_det = dets(frontal & (~non_detected),1);
ty_det = dets(frontal & (~non_detected),2);

new_widths_f = widths_det * s_width_f;
new_heights_f = heights_det * s_height_f;
new_tx_f = widths_det * s_tx_f + tx_det;
new_ty_f = heights_det * s_ty_f + ty_det;

%% Perform the profile correction
widths_det = dets(right & (~non_detected),3) - dets(right & (~non_detected),1);
heights_det = dets(right & (~non_detected),4) - dets(right & (~non_detected),2);
tx_det = dets(right & (~non_detected),1);
ty_det = dets(right & (~non_detected),2);

new_widths_p = widths_det * s_width_p;
new_heights_p = heights_det * s_height_p;
new_tx_p = widths_det * s_tx_p + tx_det;
new_ty_p = heights_det * s_ty_p + ty_det;

%% Perform the evaluation
overlaps = zeros(numel(widths_det), 1);
new_overlaps = zeros(numel(widths_det), 1);

new_bboxes = zeros(size(bboxes));
new_bboxes(frontal & (~non_detected),1) = new_tx_f;
new_bboxes(frontal & (~non_detected),2) = new_ty_f;
new_bboxes(frontal & (~non_detected),3) = new_tx_f + new_widths_f;
new_bboxes(frontal & (~non_detected),4) = new_ty_f + new_heights_f;

new_bboxes(right & (~non_detected),1) = new_tx_p;
new_bboxes(right & (~non_detected),2) = new_ty_p;
new_bboxes(right & (~non_detected),3) = new_tx_p + new_widths_p;
new_bboxes(right & (~non_detected),4) = new_ty_p + new_heights_p;

bboxes_gt = bboxes(~non_detected,:);
bboxes_det = dets(~non_detected,:);
new_bboxes = new_bboxes(~non_detected,:);

for i=1:size(bboxes_gt,1)
    bbox_gt = bboxes_gt(i,:);
    bbox_old = bboxes_det(i,:);
    overlaps(i) = overlap(bbox_gt, bbox_old);
    bbox_new = new_bboxes(i,:);
    new_overlaps(i) = overlap(bbox_gt, bbox_new);
end

fprintf('Orig - %.3f, now - %.3f\n', mean(overlaps), mean(new_overlaps));