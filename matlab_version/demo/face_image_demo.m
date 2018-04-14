clear
addpath('../PDM_helpers/');
addpath(genpath('../fitting/'));
addpath('../models/');
addpath(genpath('../face_detection'));
addpath('../CCNF/');

%% loading the patch experts
   
%[patches, pdm, clmParams, early_term_params] = Load_CECLM_general();

% An accurate CCNF (or CLNF) model
[clmParams, pdm] = Load_CLM_params_wild();
[patches] = Load_Patch_Experts( '../models/general/', 'ccnf_patches_*_general.mat', [], [], clmParams);
% A simpler (but less accurate SVR)
% [patches] = Load_Patch_Experts( '../models/general/', 'svr_patches_*_general.mat', [], [], clmParams);

clmParams.multi_modal_types  = patches(1).multi_modal_types;

% Dependencies for face detection (MatConvNet), remove if not present
setup_mconvnet;
addpath('../face_detection/mtcnn/');

%%
root_dir = '../../samples/';
images = dir([root_dir, '*.jpg']);

verbose = true;
multi_view = true;

for img=1:numel(images)
    image_orig = imread([root_dir images(img).name]);

    % MTCNN face detector
    [bboxs, det_shapes, confidences] = detect_face_mtcnn(image_orig);

    % First attempt to use the Matlab one (fastest but not as accurate, if not present use yu et al.)
    % [bboxs, det_shapes] = detect_faces(image_orig, {'cascade', 'yu'});
    % Zhu and Ramanan and Yu et al. are slower, but also more accurate 
    % and can be used when vision toolbox is unavailable
    % [bboxs, det_shapes] = detect_faces(image_orig, {'yu', 'zhu'});
    
    % The complete set that tries all three detectors starting with fastest
    % and moving onto slower ones if fastest can't detect anything
    % [bboxs, det_shapes] = detect_faces(image_orig, {'cascade', 'yu', 'zhu'});
    
    if(size(image_orig,3) == 3)
        image = rgb2gray(image_orig);
    else
        image = image_orig;
    end

    %%

    if(verbose)
        f = figure;    
        if(max(image(:)) > 1)
            imshow(double(image_orig)/255, 'Border', 'tight');
        else
            imshow(double(image_orig), 'Border', 'tight');
        end
        axis equal;
        hold on;
    end

    for i=1:size(bboxs,1)

        % Convert from the initial detected shape to CLM model parameters,
        % if shape is available
        
        bbox = bboxs(i,:);

        % have a multi-view version
        if(multi_view)

            views = [0,0,0; 0,-30,0; 0,30,0; 0,-55,0; 0,55,0; 0,0,30; 0,0,-30; 0,-90,0; 0,90,0; 0,-70,40; 0,70,-40];
            views = views * pi/180;                                                                                                                                                                     

            if(exist('early_term_params', 'var'))
                [shape,~,~,lhood,lmark_lhood,view_used] =...
                    Fitting_from_bb_multi_hyp(image, [], bbox, pdm, patches, clmParams, views, early_term_params);
            else
                [shape,~,~,lhood,lmark_lhood,view_used] =...
                    Fitting_from_bb_multi_hyp(image, [], bbox, pdm, patches, clmParams, views);
            end
        else
            [shape,~,~,lhood,lmark_lhood,view_used] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams);
        end
        
        % shape correction for matlab format
        shape = shape + 1;

        if(verbose)

            % valid points to draw (not to draw self-occluded ones)
            v_points = logical(patches(1).visibilities(view_used,:));

            try

            plot(shape(v_points,1), shape(v_points',2),'.r','MarkerSize',20);
            plot(shape(v_points,1), shape(v_points',2),'.b','MarkerSize',10);

            catch warn

            end
        end

    end
    hold off;
    
end