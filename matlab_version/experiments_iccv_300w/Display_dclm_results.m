clear

inds = 338:1026;
%% 
scrsz = get(0,'ScreenSize');
figure1 = figure('Position',[20 50 3*scrsz(3)/4 0.9*scrsz(4)]);

set(figure1,'Units','Inches');
pos = get(figure1,'Position');
set(figure1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

% Create axes
axes1 = axes('Parent',figure1,'FontSize',40,'FontName','Helvetica');

line_width = 6;
hold on;

load('../experiments_in_the_wild/results/results_wild_dclm_general.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
shapes = experiments.shapes([1:60,62:64,66:end],:,:);
labels = labels(18:end,:,:);
shapes = shapes(18:end,:,:) + 1.0;

dclm_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(dclm_error(inds));

plot(error_x, error_y, 'r', 'DisplayName', 'DCLM ', 'LineWidth',line_width);

load('../experiments_in_the_wild/results/results_wild_clnf_general_final_inner.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
shapes = experiments.shapes([1:60,62:64,66:end],:,:);
labels = labels(18:end,:,:);
% center the pixel
shapes = shapes(18:end,:,:) + 0.5;

clnf_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(clnf_error(inds));

plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);
hold on;

load('results/300W_pocr.mat');

% center the pixel
shapes_all = experiments.shapes;

pocr_error = compute_error(labels, shapes_all(:,:,:) + 1.0);

[error_x, error_y] = cummErrorCurve(pocr_error(inds));

plot(error_x, error_y, 'DisplayName', 'PO-CR', 'LineWidth',line_width);
hold on;

load('results/300W-CFSS.mat');

% center the pixel
shapes_all = zeros(68, 2, size(estimatedPose,1));
for i = 1:size(estimatedPose,1)
    
    shapes_all(:,1,i) = estimatedPose(i,1:68)';
    shapes_all(:,2,i) = estimatedPose(i,69:end)';
end
shapes_all = shapes_all([1:60,62:64,66:end],:,:);
shapes_all = shapes_all(18:end,:,:);

cfss_error = compute_error(labels, shapes_all(:,:,:) + 0.5);

[error_x, error_y] = cummErrorCurve(cfss_error(inds));

plot(error_x, error_y, 'DisplayName', 'CFSS', 'LineWidth',line_width);
hold on;

load('results/300W_sdm.mat');

% center the pixel
shapes_all = experiments.shapes;

sdm_error = compute_error(labels, shapes_all(:,:,:));
[error_x, error_y] = cummErrorCurve(sdm_error(inds));

plot(error_x, error_y, 'DisplayName', 'SDM (CVPR 13)', 'LineWidth',line_width);
hold on;

load('results/GNDPM_300W.mat');
% center the pixel
shapes_all = shapes_all(:,:,:) + 1;
labels_all = labels_all(:,:,:);

gndpm_wild_error = compute_error(labels_all, shapes_all);

[error_x, error_y] = cummErrorCurve(gndpm_wild_error(inds));

plot(error_x, error_y, 'DisplayName', 'GNDPM', 'LineWidth',line_width);
hold on;


load('results/zhu_wild.mat');
labels_all = labels_all(18:end,:,:);
shapes_all = shapes_all(18:end,:,:);

zhu_wild_error = compute_error(labels_all, shapes_all);

[error_x, error_y] = cummErrorCurve(zhu_wild_error(inds));

plot(error_x, error_y, 'c','DisplayName', 'Tree based', 'LineWidth',line_width);

% load('results/results_wild_clm.mat');
% labels = experiments.labels([1:60,62:64,66:end],:,:);
% shapes = experiments.shapes([1:60,62:64,66:end],:,:);
% labels = labels(18:end,:,:);
% % center the pixel
% shapes = shapes(18:end,:,:) + 0.5;
% 
% clm_error = compute_error( labels,  shapes);
% 
% [error_x, error_y] = cummErrorCurve(clm_error);
% 
% plot(error_x, error_y, '--b','DisplayName', 'CLM+', 'LineWidth',line_width);

load('results/drmf_wild.mat');
labels_all = labels_all(18:end,:,:);
shapes_all = shapes_all(18:end,:,:);

drmf_error = compute_error(labels_all, shapes_all);

[error_x, error_y] = cummErrorCurve(drmf_error(inds));

plot(error_x, error_y, 'k','DisplayName', 'DRMF (CVPR 13)', 'LineWidth',line_width);

set(gca,'xtick',[0:0.01:0.1])
xlim([0.01,0.07]);
xlabel('IOD normalised MSE','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
% title('Fitting in the wild without outline','FontSize',60,'FontName','Helvetica');

leg = legend('show', 'Location', 'SouthEast');
set(leg,'FontSize',30)
savefig('results/dclm-in-the-wild-no-outline.fig');
print -dpdf results/dclm-in-the-wild-clnf-no-outline.pdf
print -dpng results/dclm-300W-no-outline.png
%% 
scrsz = get(0,'ScreenSize');
figure1 = figure('Position',[20 50 3*scrsz(3)/4 0.9*scrsz(4)]);

set(figure1,'Units','Inches');
pos = get(figure1,'Position');
set(figure1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

% Create axes
axes1 = axes('Parent',figure1,'FontSize',40,'FontName','Helvetica');

line_width = 6;
hold on;

load('../experiments_in_the_wild/results/results_wild_dclm_general.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
% center the pixel
shapes = experiments.shapes([1:60,62:64,66:end],:,:) + 0.5;

dclm_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(dclm_error(inds));

plot(error_x, error_y, 'r', 'DisplayName', 'DCLM ', 'LineWidth',line_width);

load('../experiments_in_the_wild/results/results_wild_clnf_general_final_inner.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
% center the pixel
shapes = experiments.shapes([1:60,62:64,66:end],:,:) + 1;

clnf_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(clnf_error(inds));
hold on;

plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);


load('results/300W-CFSS.mat');

% center the pixel
shapes_all = zeros(68, 2, size(estimatedPose,1));
for i = 1:size(estimatedPose,1)
    
    shapes_all(:,1,i) = estimatedPose(i,1:68)';
    shapes_all(:,2,i) = estimatedPose(i,69:end)';
end

cfss_error = compute_error(labels, shapes_all(:,:,:) + 0.5);

[error_x, error_y] = cummErrorCurve(cfss_error(inds));

plot(error_x, error_y, 'DisplayName', 'CFSS', 'LineWidth',line_width);

load('results/zhu_wild.mat');

zhu_wild_error = compute_error(labels_all(:,:,:), shapes_all(:,:,:));

[error_x, error_y] = cummErrorCurve(zhu_wild_error);

plot(error_x, error_y, '.-c','DisplayName', 'Tree based', 'LineWidth',line_width);

% load('results/yu_wild.mat');

% yu_wild_error = compute_error(lmark_dets_all(:,:,:)-1, shapes_all(:,:,:));
% yu_wild_error(isnan(yu_wild_error)) = 1;
% yu_wild_error(isinf(yu_wild_error)) = 1;
% 
% [error_x, error_y] = cummErrorCurve(yu_wild_error(inds));
% 
% plot(error_x, error_y, 'xg','DisplayName', 'Yu et al.', 'LineWidth',line_width);

% load('results/results_wild_clm.mat');
% experiments(1).labels = experiments(1).labels([1:60,62:64,66:end],:,:);
% % center the pixel
% experiments(1).shapes = experiments(1).shapes([1:60,62:64,66:end],:,:) + 0.5;
% 
% clm_error = compute_error( experiments(1).labels,  experiments(1).shapes);
% 
% [error_x, error_y] = cummErrorCurve(clm_error(inds));
% 
% plot(error_x, error_y, '--b','DisplayName', 'CLM+', 'LineWidth',line_width);

load('results/drmf_wild.mat');

drmf_error = compute_error(labels_all, shapes_all);

[error_x, error_y] = cummErrorCurve(drmf_error(inds));

plot(error_x, error_y, 'k','DisplayName', 'DRMF', 'LineWidth',line_width);

set(gca,'xtick',[0.01:0.02:0.15])
xlim([0.01,0.15]);
xlabel('IOD normalised MSE','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
%title('Fitting in the wild','FontSize',60,'FontName','Helvetica');

legend('show', 'Location', 'SouthEast');

print -dpdf results/dclm-in-the-wild-comparison.pdf
savefig('results/dclm-in-the-wild-outline.fig');
print -dpng results/dclm-300W-outline.png