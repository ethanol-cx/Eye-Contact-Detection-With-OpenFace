clear

if(isunix)
    features_exe = '"../../../build/bin/FeatureExtraction"';
else
    features_exe = '"../../../x64/Release/FeatureExtraction.exe"';
end

find_FERA2011;

output_dir = 'E:\datasets\face_datasets_processed\fera2011';

% Go two levels deep
fera_dirs = dir([FERA2011_dir, '/train*']);

parfor f1=1:numel(fera_dirs)

    fera_dirs_level_2 = dir([FERA2011_dir, fera_dirs(f1).name]);
    fera_dirs_level_2 = fera_dirs_level_2(3:end);
   
    for f2=1:numel(fera_dirs_level_2)

        vid_files = dir([FERA2011_dir, fera_dirs(f1).name, '/', fera_dirs_level_2(f2).name, '/*.avi']);
        
        for v=1:numel(vid_files)
            
            input_file = [FERA2011_dir, fera_dirs(f1).name, '/', fera_dirs_level_2(f2).name, '/', vid_files(v).name];
            
            command = sprintf('%s -f "%s" -out_dir "%s" -hogalign -pdmparams', features_exe, input_file, output_dir );

            dos(command);
            
        end

    end    
    
end