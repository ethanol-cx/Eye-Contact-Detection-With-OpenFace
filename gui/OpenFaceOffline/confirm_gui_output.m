root1 = "C:\Users\Tadas Baltrusaitis\Documents\OpenFace-GUI\x64\Release\record";
root2 = "C:\Users\Tadas Baltrusaitis\Documents\OpenFace\exe\FeatureExtraction";

gui_files = dir(sprintf('%s/*.csv', root1));

for i = 1:numel(gui_files)

    table_gui = readtable(sprintf('%s/%s', root1, gui_files(1).name));
    table_console = readtable(sprintf('%s/%s', root2, gui_files(1).name));

    var_names = table_console.Properties.VariableNames;

    for v =1:numel(var_names)
       
        feat_gui = table_gui{:,var_names(v)};
        feat_console = table_console{:,var_names(v)};
        feat_diff = norm(abs(feat_gui - feat_console));
        if(feat_diff > 0.1)
            fprintf('%s error - %.3f\n', var_names{v}, feat_diff);
        end    
    end

end