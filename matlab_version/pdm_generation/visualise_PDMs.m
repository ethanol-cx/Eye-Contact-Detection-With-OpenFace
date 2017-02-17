%% Torresani visualisation of in-the wild PDM
load './Wild_data_pdm/pdm_68_aligned_wild.mat';
load('tri_68.mat', 'T');

% Visualise it
visualisePDM(M, E, V, T, 3, 3)

%% Torresani visualisation of Menpo PDM
load './menpo_pdm/pdm_68_aligned_wild.mat';
load('tri_68.mat', 'T');

% Visualise it
visualisePDM(M, E, V, T, 3, 3)
%% The multi-pie PDM visualisation
load './Wild_data_pdm/pdm_68_multi_pie.mat';
load('tri_68.mat', 'T');

% Visualise it
visualisePDM(M, E, V, T, 5, 5);