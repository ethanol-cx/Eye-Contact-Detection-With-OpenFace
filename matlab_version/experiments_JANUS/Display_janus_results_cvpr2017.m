clear

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

load('results/results_wild_dclm_general.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
shapes = experiments.shapes([1:60,62:64,66:end],:,:);
labels = labels(18:end,:,:);
% center the pixel
shapes = shapes(18:end,:,:);

clnf_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(clnf_error);

plot(error_x, error_y, 'r','DisplayName', 'DCLM', 'LineWidth',line_width);
hold on;

load('results/results_wild_clnf_general_final_inner.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
shapes = experiments.shapes([1:60,62:64,66:end],:,:);
labels = labels(18:end,:,:);
% center the pixel
shapes = shapes(18:end,:,:);

clnf_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(clnf_error);

plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);
hold on;

load('results/JANUS-CFSS.mat');

% center the pixel
shapes_all = zeros(68, 2, size(estimatedPose,1));
for i = 1:size(estimatedPose,1)
    
    shapes_all(:,1,i) = estimatedPose(i,1:68)';
    shapes_all(:,2,i) = estimatedPose(i,69:end)';
end
shapes_all = shapes_all([1:60,62:64,66:end],:,:);
shapes_all = shapes_all(18:end,:,:);

cfss_error = compute_error(labels, shapes_all(:,:,:) - 0.5);

[error_x, error_y] = cummErrorCurve(cfss_error);

plot(error_x, error_y, 'DisplayName', 'CFSS', 'LineWidth',line_width);
hold on;

load('results/JANUS_sdm.mat');

% center the pixel
shapes_all = experiments.shapes;

sdm_error = compute_error(labels, shapes_all + 0.5);

[error_x, error_y] = cummErrorCurve(sdm_error);

plot(error_x, error_y, 'DisplayName', 'SDM', 'LineWidth',line_width);
hold on;

load('results/JANUS_pocr.mat');
shapes_all = experiments.shapes;

pocr_error = compute_error(labels, shapes_all);

[error_x, error_y] = cummErrorCurve(pocr_error);

plot(error_x, error_y,'DisplayName', 'PO-CR', 'LineWidth',line_width);
hold on;

% load('results/intraface_wild_resize.mat');
% labels_all = labels_all(18:end,:,detected);
% % center the pixel
% shapes_all = shapes_all(18:end,:,detected) + 0.5;
% 
% intraface_wild_error = compute_error(labels_all, shapes_all);
% 
% [error_x, error_y] = cummErrorCurve(intraface_wild_error);
% 
% plot(error_x, error_y, '.-g','DisplayName', 'SDM (CVPR 13)', 'LineWidth',line_width);
% hold on;

% load('results/GNDPM_300W.mat');
% % center the pixel
% shapes_all = shapes_all(:,:,detected) + 1;
% labels_all = labels_all(:,:,detected);
% 
% gndpm_wild_error = compute_error(labels_all, shapes_all);
% 
% [error_x, error_y] = cummErrorCurve(gndpm_wild_error);
% 
% plot(error_x, error_y, '-.','DisplayName', 'GNDPM (CVPR 14)', 'LineWidth',line_width);
% hold on;


% load('results/zhu_wild.mat');
% labels_all = labels_all(18:end,:,detected);
% shapes_all = shapes_all(18:end,:,detected);
% 
% zhu_wild_error = compute_error(labels_all, shapes_all);
% 
% [error_x, error_y] = cummErrorCurve(zhu_wild_error);
% 
% plot(error_x, error_y, '.-c','DisplayName', 'Tree based (CVPR 12)', 'LineWidth',line_width);

% load('results/results_wild_clm.mat');
% labels = experiments.labels([1:60,62:64,66:end],:,detected);
% shapes = experiments.shapes([1:60,62:64,66:end],:,detected);
% labels = labels(18:end,:,:);
% % center the pixel
% shapes = shapes(18:end,:,:) + 0.5;
% 
% clm_error = compute_error( labels,  shapes);
% 
% [error_x, error_y] = cummErrorCurve(clm_error);
% 
% plot(error_x, error_y, '--b','DisplayName', 'CLM+', 'LineWidth',line_width);

% load('results/drmf_wild.mat');
% labels_all = labels_all(18:end,:,detected);
% shapes_all = shapes_all(18:end,:,detected);
% 
% drmf_error = compute_error(labels_all, shapes_all);
% 
% [error_x, error_y] = cummErrorCurve(drmf_error);
% 
% plot(error_x, error_y, '-.k','DisplayName', 'DRMF (CVPR 13)', 'LineWidth',line_width);

set(gca,'xtick',[0:0.02:0.10])
xlim([0.02,0.10]);
xlabel('Size normalised MSE','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
% title('Fitting in the wild without outline','FontSize',60,'FontName','Helvetica');

leg = legend('show', 'Location', 'SouthEast');
set(leg,'FontSize',30)

print -dpdf results/Janus-no-outline.pdf
print -dpng results/Janus-no-outline.png

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

load('results/results_wild_dclm_general.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
% center the pixel
shapes = experiments.shapes([1:60,62:64,66:end],:,:);

clnf_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(clnf_error);
hold on;

plot(error_x, error_y, 'r', 'DisplayName', 'DCLM', 'LineWidth',line_width);

load('results/results_wild_clnf_general_final_inner.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
% center the pixel
shapes = experiments.shapes([1:60,62:64,66:end],:,:);

clnf_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(clnf_error);
hold on;

plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);



% load('results/zhu_wild.mat');
% 
% zhu_wild_error = compute_error(labels_all(:,:,:), shapes_all(:,:,:));
% 
% [error_x, error_y] = cummErrorCurve(zhu_wild_error);
% 
% plot(error_x, error_y, '.-c','DisplayName', 'Zhu et al.', 'LineWidth',line_width);
% 
% load('results/yu_wild.mat');
% 
% yu_wild_error = compute_error(lmark_dets_all(:,:,:)-1, shapes_all(:,:,:));
% yu_wild_error(isnan(yu_wild_error)) = 1;
% yu_wild_error(isinf(yu_wild_error)) = 1;
% 
% [error_x, error_y] = cummErrorCurve(yu_wild_error);
% 
% plot(error_x, error_y, 'xg','DisplayName', 'Yu et al.', 'LineWidth',line_width);
% 
% load('results/results_wild_clm.mat');
% experiments(1).labels = experiments(1).labels([1:60,62:64,66:end],:,:);
% % center the pixel
% experiments(1).shapes = experiments(1).shapes([1:60,62:64,66:end],:,:) + 0.5;
% 
% clm_error = compute_error( experiments(1).labels,  experiments(1).shapes);
% 
% [error_x, error_y] = cummErrorCurve(clm_error);
% 
% plot(error_x, error_y, '--b','DisplayName', 'CLM+', 'LineWidth',line_width);
% 
% load('results/drmf_wild.mat');
% 
% drmf_error = compute_error(labels_all, shapes_all);
% 
% [error_x, error_y] = cummErrorCurve(drmf_error);
% 
% plot(error_x, error_y, '-.k','DisplayName', 'DRMF', 'LineWidth',line_width);

load('results/JANUS-CFSS.mat');

% center the pixel
shapes_all = zeros(68, 2, size(estimatedPose,1));
for i = 1:size(estimatedPose,1)
    
    shapes_all(:,1,i) = estimatedPose(i,1:68)';
    shapes_all(:,2,i) = estimatedPose(i,69:end)';
end
cfss_error = compute_error(labels, shapes_all(:,:,:) - 0.5);

[error_x, error_y] = cummErrorCurve(cfss_error);

plot(error_x, error_y, 'DisplayName', 'CFSS', 'LineWidth',line_width);
hold on;

set(gca,'xtick',[0:0.02:0.10])
xlim([0.02,0.10]);
xlabel('Size normalised MSE','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
%title('Fitting in the wild','FontSize',60,'FontName','Helvetica');

legend('show', 'Location', 'SouthEast');

print -dpdf results/Janus-full.pdf
print -dpng results/Janus-full.png
