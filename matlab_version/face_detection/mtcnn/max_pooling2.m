function [ output_maps ] = max_pooling2( input_maps, kernel_size, stride)
%POOLING Summary of this function goes here
%   Detailed explanation goes here
    
    orig_rows = size(input_maps,1);
    orig_cols = size(input_maps,2);
    
    pooled_rows = round((orig_rows - kernel_size)/stride) + 1;
    pooled_cols = round((orig_cols - kernel_size)/stride) + 1;

    up_to_rows_out = floor((orig_rows - kernel_size)/stride) + 1;
    up_to_cols_out = floor((orig_cols - kernel_size)/stride) + 1;

    % How many full max-pooling steps are there
    up_to_cols = kernel_size + (up_to_cols_out-1) * stride;
    up_to_rows = kernel_size + (up_to_rows_out-1) * stride;
    
    output_maps = zeros(pooled_rows, pooled_cols, size(input_maps,3), size(input_maps,4));
        
    % Pick only the striding elements
    [y, x] = meshgrid(1:up_to_cols-kernel_size+1, 1:up_to_rows-kernel_size+1);
    to_keep_map = mod(y, stride) == 1 & mod(x, stride) == 1;
    to_keep = find(to_keep_map);
    
    for m=1:size(input_maps,4)
        for i=1:size(input_maps,3)
            temp = im2col(input_maps(1:up_to_rows,1:up_to_cols,i,m), [kernel_size, kernel_size], 'sliding');        
            temp = temp(:,to_keep);
            max_val = max(temp);
            output_maps(1:up_to_rows_out,1:up_to_cols_out,i,m) = reshape(max_val, up_to_rows_out, up_to_cols_out);     
        end
    end
    % A bit of a hack for non-even number of rows or columns
    if(orig_cols ~= up_to_cols)
        span = orig_cols - (up_to_cols - kernel_size + stride);
        for m=1:size(input_maps,4)
            for i=1:size(input_maps,3)
                temp = im2col(input_maps(1:up_to_rows,end-span+1:end,i,m), [kernel_size, span], 'sliding');
                max_val = max(temp(:,1:stride:end));
                output_maps(1:up_to_rows_out,end,i,m) = max_val;     
            end        
        end
    end

    if(orig_rows ~= up_to_rows)
        span = orig_rows - (up_to_rows - kernel_size + stride);
        for m=1:size(input_maps,4)
            for i=1:size(input_maps,3)
                temp = im2col(input_maps(end-span+1:end, 1:up_to_cols,i,m), [span, kernel_size], 'sliding');
                max_val = max(temp(:,1:stride:end));
                output_maps(end, 1:up_to_cols_out,i,m) = max_val;     
            end   
        end
    end
    
    if(orig_cols ~= up_to_cols && orig_rows ~= up_to_rows)
        for m=1:size(input_maps,4)
            for i=1:size(input_maps,3)
                tmp = input_maps(up_to_rows- kernel_size + stride + 1:end,up_to_cols - kernel_size + stride+1:end,i,m);            
                output_maps(end,end,i,m) = max(tmp(:));
            end
        end
    end
        
end

