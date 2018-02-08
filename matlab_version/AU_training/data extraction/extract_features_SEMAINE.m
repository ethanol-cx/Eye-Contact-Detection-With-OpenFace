
if(isunix)
    features_exe = '"../../../build/bin/FeatureExtraction"';
else
    features_exe = '"../../../x64/Release/FeatureExtraction.exe"';
end

find_SEMAINE;


% Go two levels deep
semaine_dirs = train_recs;
output_dir = 'E:\datasets\face_datasets_processed\semaine';

parfor f1=1:numel(semaine_dirs)

    if(isdir([SEMAINE_dir, semaine_dirs{f1}]))
        
        vid_files = dir([SEMAINE_dir, semaine_dirs{f1}, '/*.avi']);

        f1_dir = semaine_dirs{f1};
        
        for v=1:numel(vid_files)

            input_file = [SEMAINE_dir, f1_dir, '/', vid_files(v).name];

            command = sprintf('%s -f "%s" -out_dir "%s" -hogalign -pdmparams', features_exe, input_file, output_dir );

            dos(command);
        end
    end
end

%%
semaine_dirs = devel_recs;
out_loc = [SEMAINE_dir, '../processed_data/devel/'];

parfor f1=1:numel(semaine_dirs)

    if(isdir([SEMAINE_dir, semaine_dirs{f1}]))
        
        vid_files = dir([SEMAINE_dir, semaine_dirs{f1}, '/*.avi']);

        f1_dir = semaine_dirs{f1};
        
        for v=1:numel(vid_files)

            input_file = [SEMAINE_dir, f1_dir, '/', vid_files(v).name];

            command = sprintf('%s -f "%s" -out_dir "%s" -hogalign -pdmparams', features_exe, input_file, output_dir );

            dos(command);

        end
    end
end