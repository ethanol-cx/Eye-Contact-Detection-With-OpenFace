function [output_dir] = run_bu_experiment(bu_dir, verbose, varargin)
   
    if(isunix)
        executable = '"../../build/bin/FeatureExtraction"';
    else
        executable = '"../../x64/Release/FeatureExtraction.exe"';
    end

    output_dir = 'experiments/bu_out/';        

    buFiles = dir([bu_dir '*.avi']);
    
    numTogether = 25;
    
    for i=1:numTogether:numel(buFiles)
        
        command = executable;
        command = cat(2, command, [' -inroot ' '"' bu_dir '/"']);
        
        % deal with edge cases
        if(numTogether + i > numel(buFiles))
            numTogether = numel(buFiles) - i + 1;
        end
        
        for n=0:numTogether-1
            inputFile = [buFiles(n+i).name];            
            command = cat(2, command, [' -f "' inputFile '" -of "' output_dir '"']);
        end
        
        % Only outputing the pose (-pose)
        command = cat(2, command,  ' -fx 500 -fy 500 -cx 160 -cy 120 -pose -vis-track ');        
        
        if(verbose)
            command = cat(2, command, [' -tracked ' outputVideo]);
        end
    
        if(any(strcmp('model', varargin)))
            command = cat(2, command, [' -mloc "', varargin{find(strcmp('model', varargin))+1}, '"']);
        end  
        
        if(isunix)
            unix(command, '-echo')
        else
            dos(command);
        end
    end
            
end