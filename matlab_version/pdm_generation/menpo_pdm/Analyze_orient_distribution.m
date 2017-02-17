clear
load('menpo_68_pts.mat');

xs = all_pts(1:end/2,:);
ys = all_pts(end/2+1:end,:);
num_imgs = size(xs, 1);

rots = zeros(3, num_imgs);
errs = zeros(1,num_imgs);

pdmLoc = ['../../models/pdm/pdm_68_aligned_wild.mat'];

load(pdmLoc);

pdm = struct;
pdm.M = double(M);
pdm.E = double(E);
pdm.V = double(V);

for i=1:num_imgs
    
    
    labels_curr = cat(2, xs(i,:)', ys(i,:)');
    labels_curr(labels_curr==-1) = 0;

    [ a, R, T, ~, l_params, err, shapeOrtho] = fit_PDM_ortho_proj_to_2D(pdm.M, pdm.E, pdm.V, labels_curr);
    errs(i) = err/a;
    rots(:,i) = Rot2Euler(R);
    
    if(errs(i) < 0 || errs(i) > 4)
       a = 2; 
    end       
end

% Current error is 2.1728 on the training data