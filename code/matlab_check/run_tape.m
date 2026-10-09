% Runs one variant of the authors' script with the tape-driven random numbers and saves the result.
addpath(fullfile(pwd, 'shadow_tape')); addpath(fullfile(pwd, 'funcs')); addpath(fullfile(pwd, 'work'));
run(getenv('GPC_SCRIPT'));
P = vertcat(pop.Position); C = [pop.Cost]';
st = [rand('__status__'); randi('__status__'); unifrnd('__status__')];
save('-v7', getenv('GPC_OUT'), 'BestCost', 'P', 'C', 'st');
