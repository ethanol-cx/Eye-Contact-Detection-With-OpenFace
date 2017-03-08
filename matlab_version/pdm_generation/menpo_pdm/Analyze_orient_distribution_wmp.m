clear
load('menpo_68_pts_valid.mat');
addpath('../PDM_helpers');

xs = all_pts(1:end/2,:);
ys = all_pts(end/2+1:end,:);
num_imgs = size(xs, 1);

rots = zeros(3, num_imgs);
errs = zeros(1,num_imgs);

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
    errs(i) = err/a;
    rots(:,i) = Rot2Euler(R);
    
    if(errs(i) < 0 || errs(i) > 4)
       fprintf('i - %d, err - %.3f\n', i, errs(i));
       errs_poss = cat(1, errs_poss, i);
    end       
end

% 300W PDM error is 1.5373
% Menpo PDM leads to 1.2140 error (100 iters, 50% rem, 25 dof, mirror, annealing 60) 
% Menpo PDM v2 leads to 1.1488 error (25 dof, 20% rem, 200 iters, annealing 60)
% Menpo PDM v3 leads to 1.1546 error (25 dof, 10% rem, 300 iters, annealing 60)
% Menpo PDM v4 leads to 1.1263 error (30 dof, no rem, 200 iters, annealing const 100)
% Menpo PDM v5 leads to 1.1312 (30 dof, no rem, 200 iters, annealing const 100)