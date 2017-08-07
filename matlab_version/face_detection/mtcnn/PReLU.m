function [ out_map ] = PReLU( input_maps, PReLU_params )
%PRELU Summary of this function goes here
%   Detailed explanation goes here

    out_map = [];
    if(numel(size(input_maps)) > 2)
        for i=1:size(input_maps,3)
            in_map = input_maps(:,:,i,:);
            in_map(in_map < 0) = in_map(in_map<0) * PReLU_params(i);
            out_map = cat(3, out_map, in_map);
        end  
    else
        for i=1:size(input_maps,2)
            in_map = input_maps(:,i);
            in_map(in_map < 0) = in_map(in_map<0) * PReLU_params(i);
            out_map = cat(2, out_map, in_map);
        end        
    end 
end

