menpo_root = 'C:\Users\tbaltrus\Documents\menpo_data_orig/';

pts_files = dir([menpo_root, '/*.jpg']);
jpg_files = dir([menpo_root, '/*.pts']);

% Split the menpo data into training (2/3rds) and validation (1/3rd)
rng(0);
indices = randperm(numel(pts_files));
train_ind = indices(1:round(2*numel(pts_files)/3));
valid_ind = indices(round(2*numel(pts_files)/3)+1:end);

% Do the actual copying to respective folders
out_train = [menpo_root, '/train/'];
out_valid = [menpo_root, '/valid/'];
mkdir(out_train);
mkdir(out_valid);

train_imgs = jpg_files(train_ind);
valid_imgs = jpg_files(valid_ind);

train_pts = pts_files(train_ind);
valid_pts = pts_files(valid_ind);

for i=1:numel(train_ind)
   
    copyfile([menpo_root, train_pts(i).name], [out_train, train_pts(i).name]);
    copyfile([menpo_root, train_imgs(i).name], [out_train, train_imgs(i).name]);
    
end

for i=1:numel(valid_ind)
   
    copyfile([menpo_root, valid_pts(i).name], [out_valid, valid_pts(i).name]);
    copyfile([menpo_root, valid_imgs(i).name], [out_valid, valid_imgs(i).name]);
    
end