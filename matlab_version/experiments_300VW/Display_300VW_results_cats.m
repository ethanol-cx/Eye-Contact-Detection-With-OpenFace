clear

scrsz = get(0,'ScreenSize');
figure1 = figure('Position',[20 50 3*scrsz(3)/4 0.9*scrsz(4)]);

set(figure1,'Units','Inches');
pos = get(figure1,'Position');
set(figure1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

% Create axes
axes1 = axes('Parent',figure1,'FontSize',40,'FontName','Helvetica');

line_width = 6;
hold on;

database_root = 'D:\Datasets\300VW_Dataset_2015_12_14\300VW_Dataset_2015_12_14/';
pocr_dir = 'D:\Dropbox\Dropbox\AAM\3rd party models\PO-CR\300VW/';
cfss_dir = 'C:\Users\tbaltrus\Documents\300W-CFSS/';
%% Gather predictions and ground truth
d_loc = '300VW_features/';
extra_dir = 'D:\Datasets\300VW_Dataset_2015_12_14\extra';

cat_1 = [ 114, 124, 125, 126, 150, 158, 401, 402, 505, 506, 507, 508, 509, 510, 511, 514, 515, 518, 519, 520, 521, 522, 524, 525, 537, 538, 540, 541, 546, 547, 548];
% cat_1 = [ 114, 124, 125, 126, 150, 158, 401, 402, 505, 506, 507, 508, 509, 510, 511, 514, 515, 518, 519, 520, 521, 522, 524, 525, 537];
cat_2 = [203, 208, 211, 212, 213, 214, 218, 224, 403, 404, 405, 406, 407, 408, 409, 412, 550, 551, 553];
cat_3 = [410, 411, 516, 517, 526, 528, 529, 530, 531, 533, 557, 558, 559, 562];

%%
load('results/cat_ids.mat');

%%

dclm_preds = zeros(68,2,0);
clnf_preds = zeros(68,2,0);
pocr_preds = zeros(49,2,0);
cfss_preds = zeros(68,2,0);
clm_preds = zeros(68,2,0);

labels = zeros(68,2,0);
% Load DCLM and CLNF results
for i=cat_1
    load(['DCLM_res/', num2str(i)]);    
    dclm_preds = cat(3, dclm_preds, preds);

    load(['CLNF_res/', num2str(i)]);    
    clnf_preds = cat(3, clnf_preds, preds);

    load(['CLM_res/', num2str(i)]);    
    clm_preds = cat(3, clm_preds, preds);

    labels = cat(3, labels, gt_landmarks);    

    load([pocr_dir,  num2str(i)]);
    pocr_preds = cat(3, pocr_preds, preds);
    
    load([cfss_dir,  num2str(i)]);
    cfss_preds = cat(3, cfss_preds, preds);    
end

dclm_preds = dclm_preds([1:60,62:64,66:end],:,:);
labels = labels([1:60,62:64,66:end],:,:);
labels = labels(18:end,:,:);
dclm_preds = dclm_preds(18:end,:,:);

dclm_error = compute_error(labels, dclm_preds);

[error_x, error_y] = cummErrorCurve(dclm_error);
plot(error_x, error_y, 'r', 'DisplayName', 'DCLM', 'LineWidth',line_width);
hold on;

clnf_preds = clnf_preds([1:60,62:64,66:end],:,:);
clnf_preds = clnf_preds(18:end,:,:);

clnf_error = compute_error(labels, clnf_preds);

[error_x, error_y] = cummErrorCurve(clnf_error);
plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);
hold on;

load('results/300VW_SDM.mat');
[error_x, error_y] = cummErrorCurve(sdm_error(inds_cat1));
plot(error_x, error_y, 'DisplayName', 'SDM', 'LineWidth',line_width);

cfss_preds = cfss_preds([1:60,62:64,66:end],:,:);
cfss_preds = cfss_preds(18:end,:,:);

cfss_error = compute_error(labels, cfss_preds - 1);

[error_x, error_y] = cummErrorCurve(cfss_error);
plot(error_x, error_y, 'DisplayName', 'CFSS', 'LineWidth',line_width);
hold on;

load('results/300VW_chehra.mat');
[error_x, error_y] = cummErrorCurve(chehra_error(inds_cat1));
plot(error_x, error_y, 'DisplayName', 'Chehra', 'LineWidth',line_width);

pocr_error = compute_error(labels, pocr_preds);
[error_x, error_y] = cummErrorCurve(pocr_error);
plot(error_x, error_y, 'DisplayName', 'PO-CR', 'LineWidth',line_width);

clm_preds = clm_preds([1:60,62:64,66:end],:,:);
clm_preds = clm_preds(18:end,:,:);

clm_error = compute_error(labels, clm_preds);

[error_x, error_y] = cummErrorCurve(clm_error);
plot(error_x, error_y, 'DisplayName', 'CLM+', 'LineWidth',line_width);
hold on;

% Make it look nice and print to a pdf
set(gca,'xtick',[0.01:0.01:0.07])
xlim([0.01,0.07]);
xlabel('IOD normalized MSE','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
ax=legend('show', 'Location', 'SouthEast');
ax.FontSize = 50;
print -dpdf results/300VWres_49_cat1.pdf

%%
dclm_preds = zeros(68,2,0);
clnf_preds = zeros(68,2,0);
pocr_preds = zeros(49,2,0);
cfss_preds = zeros(68,2,0);
clm_preds = zeros(68,2,0);

labels = zeros(68,2,0);
% Load DCLM and CLNF results
for i=cat_2
    load(['DCLM_res/', num2str(i)]);    
    dclm_preds = cat(3, dclm_preds, preds);

    load(['CLNF_res/', num2str(i)]);    
    clnf_preds = cat(3, clnf_preds, preds);

    load(['CLM_res/', num2str(i)]);    
    clm_preds = cat(3, clm_preds, preds);

    labels = cat(3, labels, gt_landmarks);    

    load([pocr_dir,  num2str(i)]);
    pocr_preds = cat(3, pocr_preds, preds);
    
    load([cfss_dir,  num2str(i)]);
    cfss_preds = cat(3, cfss_preds, preds);    
end

scrsz = get(0,'ScreenSize');
figure1 = figure('Position',[20 50 3*scrsz(3)/4 0.9*scrsz(4)]);

set(figure1,'Units','Inches');
pos = get(figure1,'Position');
set(figure1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

% Create axes
axes1 = axes('Parent',figure1,'FontSize',40,'FontName','Helvetica');

line_width = 6;
hold on;

dclm_preds = dclm_preds([1:60,62:64,66:end],:,:);
labels = labels([1:60,62:64,66:end],:,:);
labels = labels(18:end,:,:);
dclm_preds = dclm_preds(18:end,:,:);

dclm_error = compute_error(labels, dclm_preds);

[error_x, error_y] = cummErrorCurve(dclm_error);
plot(error_x, error_y, 'r', 'DisplayName', 'DCLM', 'LineWidth',line_width);
hold on;

clnf_preds = clnf_preds([1:60,62:64,66:end],:,:);
clnf_preds = clnf_preds(18:end,:,:);

clnf_error = compute_error(labels, clnf_preds);

[error_x, error_y] = cummErrorCurve(clnf_error);
plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);
hold on;

load('results/300VW_SDM.mat');
[error_x, error_y] = cummErrorCurve(sdm_error(inds_cat2));
plot(error_x, error_y, 'DisplayName', 'SDM', 'LineWidth',line_width);

cfss_preds = cfss_preds([1:60,62:64,66:end],:,:);
cfss_preds = cfss_preds(18:end,:,:);

cfss_error = compute_error(labels, cfss_preds - 1);

[error_x, error_y] = cummErrorCurve(cfss_error);
plot(error_x, error_y, 'DisplayName', 'CFSS', 'LineWidth',line_width);

load('results/300VW_chehra.mat');
[error_x, error_y] = cummErrorCurve(chehra_error(inds_cat2));
plot(error_x, error_y, 'DisplayName', 'Chehra', 'LineWidth',line_width);

pocr_error = compute_error(labels, pocr_preds);
[error_x, error_y] = cummErrorCurve(pocr_error);
plot(error_x, error_y, 'DisplayName', 'PO-CR', 'LineWidth',line_width);

clm_preds = clm_preds([1:60,62:64,66:end],:,:);
clm_preds = clm_preds(18:end,:,:);

clm_error = compute_error(labels, clm_preds);

[error_x, error_y] = cummErrorCurve(clm_error);
plot(error_x, error_y, 'DisplayName', 'CLM+', 'LineWidth',line_width);
hold on;

% Make it look nice and print to a pdf
set(gca,'xtick',[0.01:0.01:0.07])
xlim([0.01,0.07]);
xlabel('IOD normalized MSE','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
ax=legend('show', 'Location', 'SouthEast');
ax.FontSize = 50;
print -dpdf results/300VWres_49_cat2.pdf

%%
dclm_preds = zeros(68,2,0);
clnf_preds = zeros(68,2,0);
pocr_preds = zeros(49,2,0);
cfss_preds = zeros(68,2,0);
clm_preds = zeros(68,2,0);

labels = zeros(68,2,0);
% Load DCLM and CLNF results
for i=cat_3
    load(['DCLM_res/', num2str(i)]);    
    dclm_preds = cat(3, dclm_preds, preds);

    load(['CLNF_res/', num2str(i)]);    
    clnf_preds = cat(3, clnf_preds, preds);

    load(['CLM_res/', num2str(i)]);    
    clm_preds = cat(3, clm_preds, preds);

    labels = cat(3, labels, gt_landmarks);    

    load([pocr_dir,  num2str(i)]);
    pocr_preds = cat(3, pocr_preds, preds);
    
    load([cfss_dir,  num2str(i)]);
    cfss_preds = cat(3, cfss_preds, preds);    
end

scrsz = get(0,'ScreenSize');
figure1 = figure('Position',[20 50 3*scrsz(3)/4 0.9*scrsz(4)]);

set(figure1,'Units','Inches');
pos = get(figure1,'Position');
set(figure1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

% Create axes
axes1 = axes('Parent',figure1,'FontSize',40,'FontName','Helvetica');

line_width = 6;
hold on;

dclm_preds = dclm_preds([1:60,62:64,66:end],:,:);
labels = labels([1:60,62:64,66:end],:,:);
labels = labels(18:end,:,:);
dclm_preds = dclm_preds(18:end,:,:);

dclm_error = compute_error(labels, dclm_preds );

[error_x, error_y] = cummErrorCurve(dclm_error);
plot(error_x, error_y, 'r', 'DisplayName', 'DCLM', 'LineWidth',line_width);
hold on;

clnf_preds = clnf_preds([1:60,62:64,66:end],:,:);
clnf_preds = clnf_preds(18:end,:,:);

clnf_error = compute_error(labels, clnf_preds );

[error_x, error_y] = cummErrorCurve(clnf_error);
plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);
hold on;

load('results/300VW_SDM.mat');
[error_x, error_y] = cummErrorCurve(sdm_error(inds_cat3));
plot(error_x, error_y, 'DisplayName', 'SDM', 'LineWidth',line_width);

cfss_preds = cfss_preds([1:60,62:64,66:end],:,:);
cfss_preds = cfss_preds(18:end,:,:);

cfss_error = compute_error(labels, cfss_preds - 1);

[error_x, error_y] = cummErrorCurve(cfss_error);
plot(error_x, error_y, 'DisplayName', 'CFSS', 'LineWidth',line_width);

load('results/300VW_chehra.mat');
[error_x, error_y] = cummErrorCurve(chehra_error(inds_cat3));
plot(error_x, error_y, 'DisplayName', 'Chehra', 'LineWidth',line_width);

pocr_error = compute_error(labels, pocr_preds);
[error_x, error_y] = cummErrorCurve(pocr_error);
plot(error_x, error_y, 'DisplayName', 'PO-CR', 'LineWidth',line_width);

clm_preds = clm_preds([1:60,62:64,66:end],:,:);
clm_preds = clm_preds(18:end,:,:);

clm_error = compute_error(labels, clm_preds);

[error_x, error_y] = cummErrorCurve(clm_error);
plot(error_x, error_y, 'DisplayName', 'CLM+', 'LineWidth',line_width);
hold on;

% Make it look nice and print to a pdf
set(gca,'xtick',[0.01:0.01:0.07])
xlim([0.01,0.07]);
xlabel('IOD normalized MSE','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
ax=legend('show', 'Location', 'SouthEast');
ax.FontSize = 50;
print -dpdf results/300VWres_49_cat3.pdf

%%