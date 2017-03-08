clear
load('menpo_68_pts_valid.mat');
addpath('../PDM_helpers');

xs = all_pts(1:end/2,:);
ys = all_pts(end/2+1:end,:);
num_imgs = size(xs, 1);

rots_menpo = zeros(3, num_imgs);
errs_menpo = zeros(1,num_imgs);

% pdmLoc = ['../../models/pdm/pdm_68_aligned_wild.mat'];
pdmLoc = ['pdm_68_aligned_menpo_v5.mat'];

load(pdmLoc);

pdm = struct;
pdm.M = double(M);
pdm.E = double(E);
pdm.V = double(V);
errs_poss = [];
for i=1:num_imgs
    
    labels_curr = cat(2, xs(i,:)', ys(i,:)');
    labels_curr(labels_curr==-1) = 0;

    [ a, R, T, ~, l_params, err, shapeOrtho] = fit_PDM_ortho_proj_to_2D(pdm.M, pdm.E, pdm.V, labels_curr);
    errs_menpo(i) = err/a;
    rots_menpo(:,i) = Rot2Euler(R);
    
    if(errs_menpo(i) < 0 || errs_menpo(i) > 4)
       fprintf('i - %d, err - %.3f\n', i, errs_menpo(i));
       errs_poss = cat(1, errs_poss, i);
    end       
end

[~, ~, labels] = Collect_wild_imgs_train('D:/Dropbox/Dropbox/AAM/test data/');
num_imgs = size(labels,1);
rots_wild = zeros(3, num_imgs);
errs_wild = zeros(1,num_imgs);

for i=1:num_imgs
    
    labels_curr = squeeze(labels(i,:,:));
    labels_curr(labels_curr==-1) = 0;

    [ a, R, T, ~, l_params, err, shapeOrtho] = fit_PDM_ortho_proj_to_2D(pdm.M, pdm.E, pdm.V, labels_curr);
    errs_wild(i) = err/a;
    rots_wild(:,i) = Rot2Euler(R);
    
    if(errs_wild(i) < 0 || errs_wild(i) > 4)
       fprintf('i - %d, err - %.3f\n', i, errs_wild(i));
       errs_poss = cat(1, errs_poss, i);
    end       
end
errs = cat(2, errs_wild, errs_menpo);
fprintf('%.3f, %.3f, %.3f\n', mean(errs_menpo), mean(errs_wild), mean(errs));
% 300W PDM error is 1.537, 1.017, 1.285
% Menpo PDM v1 leads to 1.214, 1.184, 1.200 error (100 iters, 50% rem, 25 dof, mirror, annealing 60) 
% Menpo PDM v2 leads to 1.149, 1.138, 1.144 error (25 dof, 20% rem, 200 iters, annealing 60)
% Menpo PDM v3 leads to 1.155, 1.137, 1.146 error (25 dof, 10% rem, 300 iters, annealing 60)
% Menpo PDM v4 leads to 1.126, 1.066, 1.097 error (30 dof, no rem, 200 iters, annealing const 100)
% Menpo PDM v5 leads to 1.133, 0.986, 1.062 (30 dof, no rem, 200 iters, annealing const 100)