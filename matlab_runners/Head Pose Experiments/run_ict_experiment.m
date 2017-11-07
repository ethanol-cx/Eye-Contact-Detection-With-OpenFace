function [output_dir] = run_ict_experiment(rootDir, ictDir, verbose, varargin)
%EVALUATEICTDATABASE Summary of this function goes here
%   Detailed explanation goes here

if(isunix)
    executable = '"../../build/bin/FeatureExtraction"';
else
    executable = '"../../x64/Release/FeatureExtraction.exe"';
end

output_dir = 'experiments/ict_out';    

dbSeqDir = dir([rootDir ictDir]);
   
output_dir = cat(2, output_dir, '/');

numTogether = 10;

for i=3:numTogether:numel(dbSeqDir)
        
    command = [executable  ' -fx 535 -fy 536 -cx 327 -cy 241 -pose -vis-track '];

    command = cat(2, command, [' -inroot ' '"' rootDir '/"']);

    % deal with edge cases
    if(numTogether + i > numel(dbSeqDir))
        numTogether = numel(dbSeqDir) - i + 1;
    end
    
    for n=0:numTogether-1
        
        inputFile = [ictDir dbSeqDir(i+n).name '/colour undist.avi'];
        
        command = cat(2, command,  [' -f "' inputFile '" -of "' output_dir  '" ']);
                    
    end
    
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

