%% 
clear;

load('results/results_clnf_cross-data.mat');
load('results/menpo_labels.mat');
[clnf_error, frontal_ids] = compute_error_menpo_1( labels,  experiments.shapes);
clnf_error_frontal = clnf_error(frontal_ids);
clnf_error_profile = clnf_error(~frontal_ids);

load('results/results_ceclm_cross-data.mat');

[dclm_error, frontal_ids] = compute_error_menpo_1( labels,  experiments.shapes);
dclm_error_frontal = dclm_error(frontal_ids);
dclm_error_profile = dclm_error(~frontal_ids);

load('results/tcdcn_menpo.mat');
for i = 1:numel(shapes)
    shapes{i} = shapes{i}+0.5;
end

[tcdcn_error, frontal_ids] = compute_error_menpo_1(labels, shapes);
tcdcn_error_frontal = tcdcn_error(frontal_ids);
tcdcn_error_profile = tcdcn_error(~frontal_ids);

load('results/CFAN_menpo_train.mat');
for i = 1:numel(shapes)
    shapes{i} = shapes{i}-0.5;
end

[cfan_error, frontal_ids] = compute_error_menpo_1(labels, shapes);
cfan_error_frontal = cfan_error(frontal_ids);
cfan_error_profile = cfan_error(~frontal_ids);

load('results/menpo_train_3DDFA.mat');
for i = 1:numel(shapes)
    shapes{i} = shapes{i}-0.5;
end

[error_3ddfa, frontal_ids] = compute_error_menpo_1(labels, shapes);
error_3ddfa_frontal = error_3ddfa(frontal_ids);
error_3ddfa_profile = error_3ddfa(~frontal_ids);


load('results/Menpo-CFSS_train.mat');
shapes = cell(size(estimatedPoseFull,1),1);

for i = 1:numel(shapes)
    shape = cat(2, estimatedPoseFull(i,1:68)', estimatedPoseFull(i,69:end)');
    shapes{i} = shape-0.5;
end

[cfss_error, frontal_ids] = compute_error_menpo_1(labels, shapes);
cfss_error_frontal = cfss_error(frontal_ids);
cfss_error_profile = cfss_error(~frontal_ids);

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

[error_x, error_y] = cummErrorCurve(dclm_error_frontal);
plot(error_x, error_y, 'r', 'DisplayName', 'CE-CLM', 'LineWidth',line_width);

[error_x, error_y] = cummErrorCurve(clnf_error_frontal);
plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);

[error_x, error_y] = cummErrorCurve(cfan_error_frontal);
plot(error_x, error_y, 'DisplayName', 'CFAN', 'LineWidth',line_width);

[error_x, error_y] = cummErrorCurve(error_3ddfa_frontal);
plot(error_x, error_y, 'DisplayName', '3DDFA', 'LineWidth',line_width);

[error_x, error_y] = cummErrorCurve(cfss_error_frontal);
plot(error_x, error_y, 'DisplayName', 'CFSS', 'LineWidth',line_width);


[error_x, error_y] = cummErrorCurve(tcdcn_error_frontal);
plot(error_x, error_y, 'DisplayName', 'TCDCN', 'LineWidth',line_width);


set(gca,'xtick',[0:0.01:0.07])
xlim([0.01,0.07]);
xlabel('Size normalised MAE','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
% title('Fitting on Menpo frontal images','FontSize',60,'FontName','Helvetica');


leg = legend('show', 'Location', 'SouthEast');
set(leg,'FontSize',50)

[error_x, error_y] = cummErrorCurve(dclm_error_frontal);
plot(error_x, error_y, 'r', 'DisplayName', 'CE-CLM', 'LineWidth',line_width);

print -dpdf results/menpo-frontal_full.pdf
print -dpng results/menpo-frontal_full.png
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

[error_x, error_y] = cummErrorCurve(dclm_error_profile);
plot(error_x, error_y, 'r', 'DisplayName', 'CE-CLM', 'LineWidth',line_width);

[error_x, error_y] = cummErrorCurve(clnf_error_profile);
plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);

[error_x, error_y] = cummErrorCurve(cfan_error_profile);
plot(error_x, error_y, 'DisplayName', 'CFAN', 'LineWidth',line_width);

[error_x, error_y] = cummErrorCurve(error_3ddfa_profile);
plot(error_x, error_y, 'DisplayName', '3DDFA', 'LineWidth',line_width);

[error_x, error_y] = cummErrorCurve(cfss_error_profile);
plot(error_x, error_y, 'DisplayName', 'CFSS', 'LineWidth',line_width);


[error_x, error_y] = cummErrorCurve(tcdcn_error_profile);
plot(error_x, error_y, 'DisplayName', 'TCDCN', 'LineWidth',line_width);


set(gca,'xtick',[0.01:0.02:0.11])
xlim([0.03,0.11]);
xlabel('Size normalised MAE','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
% title('Fitting on Menpo frontal images','FontSize',60,'FontName','Helvetica');

leg = legend('show', 'Location', 'SouthEast');
set(leg,'FontSize',50)

[error_x, error_y] = cummErrorCurve(dclm_error_profile);
plot(error_x, error_y, 'r', 'DisplayName', 'CE-CLM', 'LineWidth',line_width);

print -dpdf results/menpo-profile_full.pdf
print -dpng results/menpo-profile_full.png