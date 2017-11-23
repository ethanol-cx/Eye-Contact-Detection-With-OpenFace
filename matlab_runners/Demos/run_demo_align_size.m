% A demo script that demonstrates how to process a single video file using
% OpenFace and extract and visualize all of the features

clear

% The location executable will depend on the OS
if(isunix)
    executable = '"../../build/bin/FeatureExtraction"';
else
    executable = '"../../x64/Release/FeatureExtraction.exe"';
end

% Input file
in_file = '../../samples/default.wmv';

% Where to store the output
output_dir = './processed_features/';

img_sizes = [64, 112, 224];

% This will take file after -f and output all the features to directory
% after -out_dir
command = sprintf('%s -f "%s" -out_dir "%s" -verbose -simalign', executable, in_file, output_dir);
                 
if(isunix)
    unix(command);
else
    dos(command);
end

%% Output aligned images
output_aligned_dir = sprintf('%s/%s_aligned/', output_dir, name);
img_files = dir([output_aligned_dir, '/*.bmp']);
imgs = cell(numel(img_files, 1));
for i=1:numel(img_files)
   imgs{i} = imread([ output_aligned_dir, '/', img_files(i).name]);
   imshow(imgs{i})
   drawnow
end