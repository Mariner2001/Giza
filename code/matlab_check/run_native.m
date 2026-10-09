% Runs one variant of the authors' script with Octave's own random numbers (seed GPC_SEED).
addpath(fullfile(pwd, 'shadow_native')); addpath(fullfile(pwd, 'funcs')); addpath(fullfile(pwd, 'work'));
rand('twister', str2double(getenv('GPC_SEED')));
run(getenv('GPC_SCRIPT'));
save('-v7', getenv('GPC_OUT'), 'BestCost');
