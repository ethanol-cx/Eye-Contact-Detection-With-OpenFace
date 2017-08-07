clear;

% Make sure we have the dependencies for convolution
od = cd('../../face_validation');
setup;
cd(od);

img = imread('test1.jpg');
height_orig = size(img,1);
width_orig = size(img,2);

% Everything is done in floats
img = single(img);

% Minimum face size
min_face_size = 30;

% Image pyramid scaling factor
factor = 0.709;

% Thresholds for the PNet, ONet, and RNet
threshold=[0.6 0.7 0.7];

min_dim = min([width_orig height_orig]);

% Face support region is 12x12 px, so from that can work out the largest
% scale (which is 12 / min), and work down from there to smallest scale (no smaller than
% 12x12px)
face_support = 12;
num_scales = floor(log(min_face_size / min_dim) / log(factor));
scales = (face_support / min_face_size)*factor.^(0:num_scales);

load('PNet_mlab');
load('RNet_mlab');
load('ONet_mlab');

total_bboxes = [];

% First the PNet stage on image pyramid
for s = scales
    h_pyr = ceil(height_orig * s);
    w_pyr = ceil(width_orig * s);

    % Resize the image and normalize to what MTCNN expects it to be
    im_data=(imresize(img, [h_pyr w_pyr],'bilinear')-127.5)*0.0078125;

    [ out_prob, out_correction ] = PNet( im_data, PNet_mlab );

    % Generate bounding boxes from the heatmap
    bboxes = generate_bounding_boxes(out_prob, out_correction, s, threshold(1), face_support);

    % TODO correct bboxes before running NMS?, as now lots of overlaping
    % boxes are present
    
    % Perform non maximum supression to remove reduntant bounding boxes
    pick = non_maximum_supression(bboxes, 0.5, 'Union');
    bboxes=bboxes(pick,:);
    if ~isempty(bboxes)
        total_bboxes = cat(1, total_bboxes, bboxes);
    end
end

if ~isempty(total_bboxes)
    % Non maximum supression accross bounding boxes, and their offset
    % correction
    total_bboxes = correct_bbox(total_bboxes(:,1:5), total_bboxes(:,6:end), false, true, true); 
    
end
num_bbox = size(total_bboxes,1);

% RNet stage
if num_bbox > 0
    
    proposal_imgs = zeros(24, 24, 3, num_bbox);
    for k=1:num_bbox
        
        width_target = total_bboxes(k,3) - total_bboxes(k,1) + 1;
        height_target = total_bboxes(k,4) - total_bboxes(k,2) + 1;
        
        % Work out the start and end indices in the original image
        start_x_in = max(total_bboxes(k,1), 1);
        start_y_in = max(total_bboxes(k,2), 1);
        end_x_in = min(total_bboxes(k,3), width_orig);
        end_y_in = min(total_bboxes(k,4), height_orig);
        
        % Work out the start and end indices in the target image
        start_x_out = max(-total_bboxes(k,1)+2, 1);
        start_y_out = max(-total_bboxes(k,2)+2, 1);
        end_x_out = min(width_target - (total_bboxes(k,3)-width_orig), width_target);
        end_y_out = min(height_target - (total_bboxes(k,4)-height_orig), height_target);
                
        tmp = zeros(height_target, width_target, 3);
        
        tmp(start_y_out:end_y_out,start_x_out:end_x_out,:) = ...
            img(start_y_in:end_y_in, start_x_in:end_x_in,:);
        
        proposal_imgs(:,:,:,k) = imresize(tmp, [24 24], 'bilinear');
    end
    
    % Normalize the proposal images
    proposal_imgs = (proposal_imgs - 127.5) * 0.0078125;
    
    % Apply RNet to proposal faces
    [ score, out_correction ] = RNet( proposal_imgs, RNet_mlab );
    out_correction = out_correction';

    % Find faces above the threshold
    to_keep = find(score > threshold(2));

    total_bboxes = [total_bboxes(to_keep,1:4) score(to_keep)'];
    out_correction = out_correction(to_keep,:);

    if ~isempty(total_bboxes)
        % Non maximum supression accross bounding boxes, and their offset
        % correction
        total_bboxes = correct_bbox(total_bboxes, out_correction, true, true, true); 
    end
end

num_bbox = size(total_bboxes,1);

% ONet stage
if num_bbox > 0
    
    proposal_imgs = zeros(48, 48, 3, num_bbox);
    for k=1:num_bbox
        
        width_target = total_bboxes(k,3) - total_bboxes(k,1) + 1;
        height_target = total_bboxes(k,4) - total_bboxes(k,2) + 1;
        
        % Work out the start and end indices in the original image
        start_x_in = max(total_bboxes(k,1), 1);
        start_y_in = max(total_bboxes(k,2), 1);
        end_x_in = min(total_bboxes(k,3), width_orig);
        end_y_in = min(total_bboxes(k,4), height_orig);
        
        % Work out the start and end indices in the target image
        start_x_out = max(-total_bboxes(k,1)+2, 1);
        start_y_out = max(-total_bboxes(k,2)+2, 1);
        end_x_out = min(width_target - (total_bboxes(k,3)-width_orig), width_target);
        end_y_out = min(height_target - (total_bboxes(k,4)-height_orig), height_target);
                
        tmp = zeros(height_target, width_target, 3);
        
        tmp(start_y_out:end_y_out,start_x_out:end_x_out,:) = ...
            img(start_y_in:end_y_in, start_x_in:end_x_in,:);
        
        proposal_imgs(:,:,:,k) = imresize(tmp, [48 48], 'bilinear');
    end
    
    % Normalize the proposal images
    proposal_imgs = (proposal_imgs - 127.5) * 0.0078125;
    
    % Apply ONet to proposal faces
    [ score, out_correction, lmarks ] = ONet( proposal_imgs, ONet_mlab );
    out_correction = out_correction';
    lmarks = lmarks';
    
    % Pick the final faces above the threshold
    to_keep = find(score > threshold(3));    
    lmarks = lmarks(to_keep, :);
    out_correction = out_correction(to_keep, :);
    total_bboxes = [total_bboxes(to_keep,1:4) score(to_keep)'];
    
    % Correct for the landmarks
    bbw = total_bboxes(:,3) - total_bboxes(:,1) + 1;
    bbh = total_bboxes(:,4) - total_bboxes(:,2) + 1;
    
    lmarks(:, 1:5) = bbw .* lmarks(:,1:5) + total_bboxes(:,1) - 1;
    lmarks(:, 6:10) = bbh .* lmarks(:,6:10) + total_bboxes(:,2) - 1;
    
    % Correct the bounding boxes
    if size(total_bboxes,1)>0				
        [total_bboxes, to_keep] = correct_bbox(total_bboxes, out_correction, true, false, false);
        lmarks = lmarks(to_keep, :);
    end
    
end