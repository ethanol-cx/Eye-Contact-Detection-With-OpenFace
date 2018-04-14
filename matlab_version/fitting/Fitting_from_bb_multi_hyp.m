function [ shape2D, global_params, local_params, final_lhood, landmark_lhoods, view_used, early_term ] = Fitting_from_bb_multi_hyp( Image, DepthImage, bounding_box, PDM, patches, clmParams, views, early_term_params)

    num_points = numel(PDM.M)/3;

    shapes = zeros(num_points, 2, size(views,1));
    ls = zeros(size(views,1),1);
    lmark_lhoods = zeros(num_points,size(views,1));
    views_used = zeros(size(views,1),1);
    globals = cell(size(views,1),1);
    locals = cell(size(views,1),1);

    % If early termination parameters are defined use clever multi-hypothesis setup
    % otherwise evaluate all of them in a simplistic manner
    if(nargin > 7)

        patches_small = patches(1);
        patches_large = patches(2:end);

        clmParams_small = clmParams;
        clmParams_small.numPatchIters = 1;
        clmParams_large = clmParams;
        clmParams_large.regFactor = clmParams_large.regFactor(2:end);
        clmParams_large.sigmaMeanShift = clmParams_large.sigmaMeanShift(2:end);
        clmParams_large.tikhonov_factor = clmParams_large.tikhonov_factor(2:end);
        clmParams_large.numPatchIters = 3;
        clmParams_large.window_size = clmParams_large.window_size(2:end,:);


        early_term = 0;
        % Find the best orientation
        for v = 1:size(views,1)
            [shapes(:,:,v),globals{v},locals{v},ls(v),lmark_lhoods(:,v),views_used(v)]...
                = Fitting_from_bb(Image, DepthImage, bounding_box, PDM, patches_small, clmParams_small, 'orientation', views(v,:));                                            

            ls(v) = ls(v) * early_term_params.weights_scale(views_used(v)) + early_term_params.weights_add(views_used(v)); 

            if(ls(v) > early_term_params.cutoffs(views_used(v)))
                early_term = v;
                [shape,global_params,local_params,final_lhood,landmark_lhoods,view_used]...
                    = Fitting_from_bb(Image, DepthImage, bounding_box, PDM, patches_large, clmParams_large, 'gparam', globals{v}, 'lparam', locals{v});
                break;
            end
        end

        if(early_term == 0)

            % Discard the really unreliable starts            
            [~, ids] = sort(ls);
            to_refine = ids(end-3:end);

            % If there are any starts left try refining them in a
            % multiscale approach
            if(~isempty(to_refine))
                shapes = zeros(num_points, 2, numel(to_refine));
                ls = zeros(numel(to_refine),1);
                lmark_lhoods = zeros(num_points,numel(to_refine));
                views_used = zeros(numel(to_refine),1);  
                globals_n = cell(size(to_refine,1),1);
                locals_n = cell(size(to_refine,1),1);            
                ind = 1;
                for v = to_refine'                    
                    [shapes(:,:,ind), globals_n{ind}, locals_n{ind},...
                        ls(ind),lmark_lhoods(:,ind),views_used(ind)] ...
                        = Fitting_from_bb(Image, DepthImage, bounding_box, PDM, patches_large, clmParams_large, 'gparam', globals{v}, 'lparam', locals{v});
                    ind = ind+1;
                end            
            end

            % TODO aggregation?
            [final_lhood, v_ind] = max(ls);
            landmark_lhoods = lmark_lhoods(:,v_ind);
            global_params = globals_n{v_ind};
            local_params = locals_n{v_ind};
            shape = shapes(:,:,v_ind);
            view_used = views_used(v_ind);
        end
        shape2D = shape;
    else

        % Find the best orientation
        for v = 1:size(views,1)
            [shapes(:,:,v),globals{v},locals{v},ls(v),lmark_lhoods(:,v),views_used(v)] = Fitting_from_bb(Image, [], bounding_box, PDM, patches, clmParams, 'orientation', views(v,:));                                            
        end

        [final_lhood, v_ind] = max(ls);
        landmark_lhoods = lmark_lhoods(:,v_ind);

        global_params = globals{v_ind};
        local_params = locals{v_ind};

        shape2D = shapes(:,:,v_ind);
        view_used = views_used(v);

    end
    
    % Check if a hierarchical model is defined, and if it is apply it here
    if(isfield(clmParams, 'hierarch'))
        % Perform hierarchical model fitting
        %, TODO this should be a loop
        shape_hierarch = shape2D(clmParams.hierarch.inds,:);

        [ a, R, T, ~, l_params, err] = fit_PDM_ortho_proj_to_2D_no_reg(clmParams.hierarch.pdm.M, clmParams.hierarch.pdm.E, clmParams.hierarch.pdm.V, shape_hierarch);
        if(a > 0.9)
            g_param = [a; Rot2Euler(R)'; T];

            bbox_2 = [min(shape_hierarch(:,1)), min(shape_hierarch(:,2)), max(shape_hierarch(:,1)), max(shape_hierarch(:,2))];

            [shape_hierarch] = Fitting_from_bb(Image, [], bbox_2, clmParams.hierarch.pdm, clmParams.hierarch.patches, clmParams.hierarch.clmParams, 'gparam', g_param, 'lparam', l_params);

            % Now after detections incorporate the subset of the landmarks into the main model
            shape2D(clmParams.hierarch.inds, :) = shape_hierarch;

            [ ~, ~, ~, ~, ~, ~, shape2D] = fit_PDM_ortho_proj_to_2D_no_reg(PDM.M, PDM.E, PDM.V, shape2D);    

        end    

    end
    
end