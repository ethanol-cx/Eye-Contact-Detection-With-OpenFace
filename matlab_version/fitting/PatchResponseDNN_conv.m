function [ responses ] = PatchResponseDNN_conv(patches, patch_experts_class, visibilities, patchExperts, window_size)
%PATCHRESPONSESVM Summary of this function goes here
%   Detailed explanation goes here

    normalisationOptions = patchExperts.normalisationOptionsCol;
    patchSize = normalisationOptions.patchSize;
                  
    responses = cell(size(patches, 1), 1);
    empty = zeros(window_size(1)-patchSize(1)+1, window_size(2)-patchSize(2)+1);
    
    for i = 1:numel(patches(:,1))
        responses{i} = empty;
        if visibilities(i)
                        
            col_norm = normalisationOptions.useNormalisedCrossCorr == 1;

            smallRegionVec = patches(i,:);
            smallRegion = reshape(smallRegionVec, window_size(1), window_size(2));
           
            % Normalize the weights
            weights = patch_experts_class{i};
            weights_normed = weights{1}(2:end,:);
            offset = repmat(mean(weights_normed), 121, 1);
            %scaling = (weights_normed - offset);
            scaling = repmat(sqrt(sum((weights_normed - offset).^2)), 121, 1);

            weights_normed2 = weights_normed;
            biases =  weights{1}(1,:);
            patch_normed = zeros(numel(biases), numel(empty));
            
            for l=1:numel(biases)
%                 pe1 = double(reshape(weights_normed1(:,l), patchSize));
                pe2 = double(reshape(weights_normed2(:,l), patchSize));
%                 r1 = conv_response(smallRegion, pe1, col_norm, patchSize) * scaling(1,l) + biases(l) + weights{2}(l);               
                r2 = conv_response(smallRegion, pe2, col_norm, patchSize) * scaling(1,l) + biases(l) + weights{2}(l);               
                patch_normed(l,:) = r2(:);
            end
            
            patch_normed = max(0, patch_normed);
            
            % Where DNN will happen
            for w =2:numel(weights)/2
                
                % mult and bias
                patch_normed = weights{(w-1)*2+1}' * patch_normed + repmat(weights{(w-1)*2+2}', 1, size(patch_normed,2));

                if w < 3
%                    patch_normed(patch_normed < 0) = 0;
                   patch_normed = max(0, patch_normed);
                else
                   patch_normed = 1./(1+exp(-patch_normed));
                end

            end
            
            responses{i}(:) = reshape(patch_normed', window_size(1)-patchSize(1)+1, window_size(2)-patchSize(2)+1);
            
        end
    end
    
end

function response = conv_response(region, patchExpert, normalise_x_corr,patchSize)

    if(normalise_x_corr)
        
        % the fast mex convolution
        [response] = normxcorr2_mex(patchExpert, region);

        response = response(patchSize(1):end-patchSize(1)+1,patchSize(2):end-patchSize(2)+1);       
    else
        % this assumes that the patch is already normed
        template = rot90(patchExpert,2);
        response = conv2(region, template, 'valid');  
    end
end
