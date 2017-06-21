clear

if(isunix)
    executable = '"../../build/bin/FeatureExtraction"';
else
    executable = '"../../x64/Release/FeatureExtraction.exe"';
end

output = 'yt_features_ceclm/';

if(~exist(output, 'file'))
    mkdir(output)
end
    
if(exist([getenv('USERPROFILE') '/Dropbox/AAM/test data/'], 'file'))
    database_root = [getenv('USERPROFILE') '/Dropbox/AAM/test data/'];    
elseif(exist('D:/Dropbox/Dropbox/AAM/test data/', 'file'))
    database_root = 'D:/Dropbox/Dropbox/AAM/test data/';
else
    database_root = '/multicomp/datasets/';
end

database_root = [database_root, '/ytceleb/'];

in_vids = dir([database_root '/*.avi']);

command = executable;
command = cat(2, command, ' -no3Dfp -noMparams -noPose -noGaze -noAUs ');
% add all videos to single argument list (so as not to load the model anew
% for every video)
for i=1:numel(in_vids)
    
    [~, name, ~] = fileparts(in_vids(i).name);
    
    % where to output tracking results
    outputFile_fp = [output name '_fp.txt'];
    in_file_name = [database_root, '/', in_vids(i).name];        
    
    command = cat(2, command, [' -f "' in_file_name '" -of "' outputFile_fp '"']);                     
end

if(isunix)
    unix(command, '-echo')
else
    dos(command);
end

%%
output = 'yt_features_clnf/';

if(~exist(output, 'file'))
    mkdir(output)
end
    
command = executable;
command = cat(2, command, ' -mloc model/main_clnf_general.txt ');
command = cat(2, command, ' -no3Dfp -noMparams -noPose -noGaze -noAUs ');

% add all videos to single argument list (so as not to load the model anew
% for every video)
for i=1:numel(in_vids)
    
    [~, name, ~] = fileparts(in_vids(i).name);
    
    % where to output tracking results
    outputFile_fp = [output name '_fp.txt'];
    in_file_name = [database_root, '/', in_vids(i).name];        
    
    command = cat(2, command, [' -f "' in_file_name '" -of "' outputFile_fp '"']);                     
end

if(isunix)
    unix(command, '-echo')
else
    dos(command);
end
%%
output = 'yt_features_clm/';

if(~exist(output, 'file'))
    mkdir(output)
end
    
command = executable;
command = cat(2, command, ' -mloc model/main_clm_general.txt ');
command = cat(2, command, ' -no3Dfp -noMparams -noPose -noGaze -noAUs ');

% add all videos to single argument list (so as not to load the model anew
% for every video)
for i=1:numel(in_vids)
    
    [~, name, ~] = fileparts(in_vids(i).name);
    
    % where to output tracking results
    outputFile_fp = [output name '_fp.txt'];
    in_file_name = [database_root, '/', in_vids(i).name];        
    
    command = cat(2, command, [' -f "' in_file_name '" -of "' outputFile_fp '"']);                     
end

if(isunix)
    unix(command, '-echo')
else
    dos(command);
end
%% evaluating yt datasets
d_loc_ceclm = 'yt_features_ceclm/';
d_loc_clnf = 'yt_features_clnf/';
d_loc_clm = 'yt_features_clm/';

files_yt = dir([d_loc_ceclm, '/*.txt']);
preds_all_ceclm = [];
preds_all_clnf = [];
preds_all_clm = [];
gts_all = [];
for i = 1:numel(files_yt)
    [~, name, ~] = fileparts(files_yt(i).name);
    pred_landmarks_ceclm = dlmread([d_loc_ceclm, files_yt(i).name], ',', 1, 0);
    pred_landmarks_ceclm = pred_landmarks_ceclm(:,5:end);
    
    xs = pred_landmarks_ceclm(:, 1:end/2);
    ys = pred_landmarks_ceclm(:, end/2+1:end);
    pred_landmarks_ceclm = zeros([size(xs,2), 2, size(xs,1)]);
    pred_landmarks_ceclm(:,1,:) = xs';
    pred_landmarks_ceclm(:,2,:) = ys';
    
    pred_landmarks_clnf = dlmread([d_loc_clnf, files_yt(i).name], ',', 1, 0);
    pred_landmarks_clnf = pred_landmarks_clnf(:,5:end);
    
    xs = pred_landmarks_clnf(:, 1:end/2);
    ys = pred_landmarks_clnf(:, end/2+1:end);
    pred_landmarks_clnf = zeros([size(xs,2), 2, size(xs,1)]);
    pred_landmarks_clnf(:,1,:) = xs';
    pred_landmarks_clnf(:,2,:) = ys';    
    
    pred_landmarks_clm = dlmread([d_loc_clm, files_yt(i).name], ',', 1, 0);
    pred_landmarks_clm = pred_landmarks_clm(:,5:end);
    
    xs = pred_landmarks_clm(:, 1:end/2);
    ys = pred_landmarks_clm(:, end/2+1:end);
    pred_landmarks_clm = zeros([size(xs,2), 2, size(xs,1)]);
    pred_landmarks_clm(:,1,:) = xs';
    pred_landmarks_clm(:,2,:) = ys';    
    
    load([database_root, name(1:end-3), '.mat']);
    preds_all_ceclm = cat(3, preds_all_ceclm, pred_landmarks_ceclm);
    preds_all_clnf = cat(3, preds_all_clnf, pred_landmarks_clnf);
    preds_all_clm = cat(3, preds_all_clm, pred_landmarks_clm);
    gts_all = cat(3, gts_all, labels);
end

%%
[ceclm_error, err_pp_ceclm] = compute_error( gts_all - 1.5,  preds_all_ceclm);
[clnf_error, err_pp_clnf] = compute_error( gts_all - 1.5,  preds_all_clnf);
[clm_error, err_pp_clm] = compute_error( gts_all - 1.5,  preds_all_clm);

filename = sprintf('results/fps_yt');
save(filename);

% Also save them in a reasonable .txt format for easy comparison
f = fopen('results/fps_yt.txt', 'w');
fprintf(f, 'Model, mean,  median\n');
fprintf(f, 'OpenFace (CE-CLM):  %.4f,   %.4f\n', mean(ceclm_error), median(ceclm_error));
fprintf(f, 'OpenFace (CLNF):  %.4f,   %.4f\n', mean(clnf_error), median(clnf_error));
fprintf(f, 'CLM:   %.4f,   %.4f\n', mean(clm_error), median(clm_error));

fclose(f);
clear 'f'