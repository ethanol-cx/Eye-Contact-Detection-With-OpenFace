clear

executable = '"../../x64/Release/FeatureExtraction.exe"';

output = '300VW_features/';

if(~exist(output, 'file'))
    mkdir(output)
end
    
if(exist('D:\Datasets\300VW_Dataset_2015_12_14\300VW_Dataset_2015_12_14/', 'file'))
    database_root = 'D:\Datasets\300VW_Dataset_2015_12_14\300VW_Dataset_2015_12_14/';    
elseif(exist('E:\datasets\300VW\300VW_Dataset_2015_12_14', 'file'))
    database_root = 'E:\datasets\300VW\300VW_Dataset_2015_12_14';
elseif(exist('/multicomp/datasets/300VW_Dataset_2015_12_14/', 'file'))
    database_root = '/multicomp/datasets/300VW_Dataset_2015_12_14/';
else
    fprintf('Could not find the dataset');
    return;
end
%%
cat_1 = [ 114, 124, 125, 126, 150, 158, 401, 402, 505, 506, 507, 508, 509, 510, 511, 514, 515, 518, 519, 520, 521, 522, 524, 525, 537, 538, 540, 541, 546, 547, 548];
cat_2 = [203, 208, 211, 212, 213, 214, 218, 224, 403, 404, 405, 406, 407, 408, 409, 412, 550, 551, 553];
cat_3 = [410, 411, 516, 517, 526, 528, 529, 530, 531, 533, 557, 558, 559, 562];
in_dirs = cat(2, cat_1, cat_2, cat_3);

%% Running CE-CLM models
parfor i=1:numel(in_dirs)
    command = executable;

    command = cat(2, command, ' -no3Dfp -noMparams -noPose -noGaze -noAUs ');
    command = cat(2, command, [' -inroot "' database_root '" ']);

    name = num2str(in_dirs(i));
    
    % where to output tracking results
    outputFile_fp = [output '/ceclm/' name '_fp.txt'];
    outputFile_vid = [output '/ceclm/' name '.avi'];
    in_file_name = ['/', name, '/vid.avi'];        
    
    command = cat(2, command, [' -f "' in_file_name '" -of "' outputFile_fp '" -ov "' outputFile_vid '"']);                     
    dos(command);
end

%% Running CLNF models
parfor i=1:numel(in_dirs)
    command = executable;

    command = cat(2, command, ' -no3Dfp -noMparams -noPose -noGaze -noAUs -mloc model/main_clnf_general.txt ');
    command = cat(2, command, [' -inroot "' database_root '" ']);

    name = num2str(in_dirs(i));
    
    % where to output tracking results
    outputFile_fp = [output '/clnf/' name '_fp.txt'];
    outputFile_vid = [output '/clnf/' name '.avi'];
    in_file_name = ['/', name, '/vid.avi'];        
    
    command = cat(2, command, [' -f "' in_file_name '" -of "' outputFile_fp '" -ov "' outputFile_vid '"']);                     
    dos(command);
end

%%
Compute_300VW_errors;

%%
Construct_error_table_300VW;

%%
Display_300VW_results_49;
Display_300VW_results_66;