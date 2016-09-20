clear

executable = '"../../x64/Release/FeatureExtraction.exe"';

output = '300VW_features/';

if(~exist(output, 'file'))
    mkdir(output)
end
    
database_root = 'D:\Datasets\300VW_Dataset_2015_12_14\300VW_Dataset_2015_12_14/';

%%
in_dirs = dir(database_root);
in_dirs = in_dirs(3:end);

parfor i=1:numel(in_dirs)
    command = executable;

    command = cat(2, command, ' -no3Dfp -noMparams -noPose -noGaze -noAUs ');
    command = cat(2, command, [' -inroot "' database_root '" ']);

    [~, name, ~] = fileparts(in_dirs(i).name);
    
    % where to output tracking results
    outputFile_fp = [output name '_fp.txt'];
    outputFile_vid = [output name '.avi'];
    in_file_name = ['/', in_dirs(i).name, '/vid.avi'];        
    
    command = cat(2, command, [' -f "' in_file_name '" -of "' outputFile_fp '" -ov "' outputFile_vid '"']);                     
    dos(command);
end

%% Gather predictions and ground truth
d_loc = '300VW_features/';
extra_dir = 'D:\Datasets\300VW_Dataset_2015_12_14\extra';


files_yt = dir([d_loc, '/*.txt']);
preds_all = [];
gts_all = [];
confs = [];
for i = 1:numel(files_yt)
    [~, name, ~] = fileparts(files_yt(i).name);
    pred_landmarks = dlmread([d_loc, files_yt(i).name], ',', 1, 0);
    conf = pred_landmarks(:,3);
    pred_landmarks = pred_landmarks(:,5:end);
    
    xs = pred_landmarks(:, 1:end/2);
    ys = pred_landmarks(:, end/2+1:end);
    pred_landmarks = zeros([size(xs,2), 2, size(xs,1)]);
    pred_landmarks(:,1,:) = xs';
    pred_landmarks(:,2,:) = ys';
       
    name = name(1:end-3);
    fps_all = dir([database_root, '/', name, '/annot/*.pts']);
    gt_landmarks = zeros(size(pred_landmarks));
    for k = 1:size(fps_all)
        
        gt_landmarks_frame = dlmread([database_root, '/', name, '/annot/', fps_all(k).name], ' ', 'A4..B71');
        gt_landmarks(:,:,k) = gt_landmarks_frame;
    end
    
    if(size(pred_landmarks,3) ~= size(fps_all))
        fprintf('something wrong at vid %s, fps - %d, preds - %d\n', name, gt_landmarks);
    end
    
    % Remove unreliable frames
    if(exist([extra_dir, '/', name, '.mat'], 'file'))
        load([extra_dir, '/', name, '.mat']);
        gt_landmarks(:,:,int32(error)) = [];
        pred_landmarks(:,:,int32(error))=[];
        conf(int32(error)) = [];
    end

	preds_all = cat(3, preds_all, pred_landmarks);
    gts_all = cat(3, gts_all, gt_landmarks);
    confs = cat(1, confs, conf);
end
%%
[clnf_error, err_pp_clnf] = compute_error( gts_all - 0.5,  preds_all);
preds_all_49 = preds_all;
preds_all_49(65,:,:) = [];
preds_all_49(61,:,:) = [];
preds_all_49 = preds_all_49(18:end,:,:);

gts_all_49 = gts_all;
gts_all_49(65,:,:) = [];
gts_all_49(61,:,:) = [];
gts_all_49 = gts_all_49(18:end,:,:);

[clnf_error49, err_pp_clnf49] = compute_error( gts_all_49 - 0.5,  preds_all_49);

filename = sprintf('results/300VW_CLNF');
save(filename, 'clnf_error', 'clnf_error49');

%%
line_width = 2;

[error_x, error_y] = cummErrorCurve(clnf_error49);
plot(error_x, error_y, 'DisplayName', 'OpenFace-49', 'LineWidth',line_width);
hold on;

load('results/300VW_SDM.mat');
[error_x, error_y] = cummErrorCurve(sdm_error);
plot(error_x, error_y, 'DisplayName', 'SDM-49', 'LineWidth',line_width);

load('results/300VW_chehra.mat');
[error_x, error_y] = cummErrorCurve(chehra_error);
plot(error_x, error_y, 'DisplayName', 'Chehra-49', 'LineWidth',line_width);

load('results/300VW_pocr.mat');
[error_x, error_y] = cummErrorCurve(pocr_error);
plot(error_x, error_y, 'DisplayName', 'PO-CR-49', 'LineWidth',line_width);

% Make it look nice and print to a pdf
set(gca,'xtick',[0:0.01:0.08])
xlim([0,0.08]);
xlabel('Size normalised shape RMS error of 49 landmarks','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
legend('show', 'Location', 'SouthEast');
print -dpdf results/300VWres_49_114vids.pdf

%%
line_width = 2;
[error_x, error_y] = cummErrorCurve(clnf_error);
plot(error_x, error_y, 'DisplayName', 'OpenFace-68', 'LineWidth',line_width);
hold on;

% Make it look nice and print to a pdf
set(gca,'xtick',[0:0.01:0.08])
xlim([0,0.08]);
xlabel('Size normalised shape RMS error of 68 landmarks','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
legend('show', 'Location', 'SouthEast');
print -dpdf results/300VWres_68_114vids.pdf

%%
% Also save them in a reasonable .txt format for easy comparison
f = fopen('results/300VW.txt', 'w');
fprintf(f, 'Model 68, mean,  median\n');
fprintf(f, 'OpenFace (CLNF):  %.4f,   %.4f\n', mean(clnf_error), median(clnf_error));
fprintf(f, '--------------------------------\n');
fprintf(f, 'Model 49/51, mean,  median\n');
fprintf(f, 'OpenFace (CLNF):  %.4f,   %.4f\n', mean(clnf_error49), median(clnf_error49));
fprintf(f, 'Intraface (SDM):  %.4f,   %.4f\n', mean(sdm_error), median(sdm_error));
fprintf(f, 'Chehra (DRMF):  %.4f,   %.4f\n', mean(chehra_error), median(chehra_error));
fprintf(f, 'PO-CR:  %.4f,   %.4f\n', mean(pocr_error), median(pocr_error));

fclose(f);
clear 'f'
