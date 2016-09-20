clear

output = '300VW_features/';
    
database_root = 'D:\Datasets\300VW_Dataset_2015_12_14\300VW_Dataset_2015_12_14/';


%% Gather predictions and ground truth
d_loc = '300VW_features/';
extra_dir = 'D:\Datasets\300VW_Dataset_2015_12_14\extra';

cat_1 = [ 114, 124, 125, 126, 150, 158, 401, 402, 505, 506, 507, 508, 509, 510, 511, 514, 515, 518, 519, 520, 521, 522, 524, 525, 537, 538, 540, 541, 546, 547, 548];
cat_2 = [203, 208, 211, 212, 213, 214, 218, 224, 403, 404, 405, 406, 407, 408, 409, 412, 550, 551, 553];
cat_3 = [410, 411, 516, 517, 526, 528, 529, 530, 531, 533, 557, 558, 559, 562];
files_yt = dir([d_loc, '/*.txt']);
preds_all = zeros([68,2,0]);
gts_all = [];
confs = [];

inds_cat1 = false(216470,1);
inds_cat2 = false(216470,1);
inds_cat3 = false(216470,1);
vid_ids = zeros(216470, 1);

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
    name_id = str2num(name);
           
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

    if(sum(cat_1 == name_id) > 0)
        inds_cat1(size(preds_all,3)+1:size(preds_all,3)+size(pred_landmarks,3)) = true;
    end
    if(sum(cat_2 == name_id) > 0)
        inds_cat2(size(preds_all,3)+1:size(preds_all,3)+size(pred_landmarks,3)) = true;
    end
    if(sum(cat_3 == name_id) > 0)
        inds_cat3(size(preds_all,3)+1:size(preds_all,3)+size(pred_landmarks,3)) = true;
    end  
    vid_ids(size(preds_all,3)+1:size(preds_all,3)+size(pred_landmarks,3)) = name_id;
	preds_all = cat(3, preds_all, pred_landmarks);
    gts_all = cat(3, gts_all, gt_landmarks);
    confs = cat(1, confs, conf);
    
end
save('results/cat_ids', 'inds_cat1', 'inds_cat2', 'inds_cat3', 'vid_ids');
%%
load('results/cat_ids.mat');

%%
load('results/300VW_CLNF.mat')
line_width = 2;

[error_x, error_y] = cummErrorCurve(clnf_error49(inds_cat1));
plot(error_x, error_y, 'DisplayName', 'OpenFace-49', 'LineWidth',line_width);
hold on;

load('results/300VW_SDM.mat');
[error_x, error_y] = cummErrorCurve(sdm_error(inds_cat1));
plot(error_x, error_y, 'DisplayName', 'SDM-49', 'LineWidth',line_width);

load('results/300VW_chehra.mat');
[error_x, error_y] = cummErrorCurve(chehra_error(inds_cat1));
plot(error_x, error_y, 'DisplayName', 'Chehra-49', 'LineWidth',line_width);

load('results/300VW_pocr.mat');
[error_x, error_y] = cummErrorCurve(pocr_error(inds_cat1));
plot(error_x, error_y, 'DisplayName', 'PO-CR-49', 'LineWidth',line_width);

% Make it look nice and print to a pdf
set(gca,'xtick',[0:0.01:0.08])
xlim([0,0.08]);
xlabel('Size normalised shape RMS error of 49 landmarks on Category 1','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
legend('show', 'Location', 'SouthEast');
print -dpdf results/300VWres_49_cat1.pdf

%%
figure
load('results/300VW_CLNF.mat')
line_width = 2;

[error_x, error_y] = cummErrorCurve(clnf_error49(inds_cat2));
plot(error_x, error_y, 'DisplayName', 'OpenFace-49', 'LineWidth',line_width);
hold on;

load('results/300VW_SDM.mat');
[error_x, error_y] = cummErrorCurve(sdm_error(inds_cat2));
plot(error_x, error_y, 'DisplayName', 'SDM-49', 'LineWidth',line_width);

load('results/300VW_chehra.mat');
[error_x, error_y] = cummErrorCurve(chehra_error(inds_cat2));
plot(error_x, error_y, 'DisplayName', 'Chehra-49', 'LineWidth',line_width);

load('results/300VW_pocr.mat');
[error_x, error_y] = cummErrorCurve(pocr_error(inds_cat2));
plot(error_x, error_y, 'DisplayName', 'PO-CR-49', 'LineWidth',line_width);

% Make it look nice and print to a pdf
set(gca,'xtick',[0:0.01:0.08])
xlim([0,0.08]);
xlabel('Size normalised shape RMS error of 49 landmarks on Category 2','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
legend('show', 'Location', 'SouthEast');
print -dpdf results/300VWres_49_cat2.pdf

%%
figure
load('results/300VW_CLNF.mat')
line_width = 2;

[error_x, error_y] = cummErrorCurve(clnf_error49(inds_cat3));
plot(error_x, error_y, 'DisplayName', 'OpenFace-49', 'LineWidth',line_width);
hold on;

load('results/300VW_SDM.mat');
[error_x, error_y] = cummErrorCurve(sdm_error(inds_cat3));
plot(error_x, error_y, 'DisplayName', 'SDM-49', 'LineWidth',line_width);

load('results/300VW_chehra.mat');
[error_x, error_y] = cummErrorCurve(chehra_error(inds_cat3));
plot(error_x, error_y, 'DisplayName', 'Chehra-49', 'LineWidth',line_width);

load('results/300VW_pocr.mat');
[error_x, error_y] = cummErrorCurve(pocr_error(inds_cat3));
plot(error_x, error_y, 'DisplayName', 'PO-CR-49', 'LineWidth',line_width);

% Make it look nice and print to a pdf
set(gca,'xtick',[0:0.01:0.08])
xlim([0,0.08]);
xlabel('Size normalised shape RMS error of 49 landmarks on Category 3','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
legend('show', 'Location', 'SouthEast');
print -dpdf results/300VWres_49_cat3.pdf

%%
sdm_err_cat_1 = sdm_error(inds_cat1);
sdm_err_cat_2 = sdm_error(inds_cat2);
sdm_err_cat_3 = sdm_error(inds_cat3);

pocr_err_cat_1 = pocr_error(inds_cat1);
pocr_err_cat_2 = pocr_error(inds_cat2);
pocr_err_cat_3 = pocr_error(inds_cat3);

chehra_err_cat_1 = chehra_error(inds_cat1);
chehra_err_cat_2 = chehra_error(inds_cat2);
chehra_err_cat_3 = chehra_error(inds_cat3);

clnf_err_cat_1 = clnf_error49(inds_cat1);
clnf_err_cat_2 = clnf_error49(inds_cat2);
clnf_err_cat_3 = clnf_error49(inds_cat3);

save('results/category_errors', 'sdm_err_cat_1', 'sdm_err_cat_2', 'sdm_err_cat_3', ...
    'pocr_err_cat_1', 'pocr_err_cat_2', 'pocr_err_cat_3', ...
    'chehra_err_cat_1', 'chehra_err_cat_2', 'chehra_err_cat_3', ...
    'clnf_err_cat_1', 'clnf_err_cat_2', 'clnf_err_cat_3')