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

load('results/results_wild_clnf_general_final_inner.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
shapes = experiments.shapes([1:60,62:64,66:end],:,:);
labels = labels(18:end,:,:) - 0.5;
shapes = shapes(18:end,:,:);

clnf_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(clnf_error);
hold on;

plot(error_x, error_y, 'DisplayName', 'CLNF', 'LineWidth',line_width);

load('results/results_wild_dclm_general.mat');
labels = experiments.labels([1:60,62:64,66:end],:,:);
shapes = experiments.shapes([1:60,62:64,66:end],:,:);
labels = labels(18:end,:,:) - 0.5;
shapes = shapes(18:end,:,:);

clm_error = compute_error( labels,  shapes);

[error_x, error_y] = cummErrorCurve(clm_error);

plot(error_x, error_y, 'DisplayName', 'DCLM', 'LineWidth',line_width);

set(gca,'xtick',[0:0.01:0.08])
xlim([0,0.08]);
xlabel('Size normalised shape RMS error','FontName','Helvetica');
ylabel('Proportion of images','FontName','Helvetica');
grid on
title('Fitting on 300W','FontSize',60,'FontName','Helvetica');

legend('show', 'Location', 'SouthEast');
print -dpng results/dclm_results.png
