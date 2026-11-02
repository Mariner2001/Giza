% Runs a variant of GPC.m with Octave's own random numbers (seed from GPC_SEED).
% Environment: GPC_VARIANT (script file), GPC_SEED, GPC_OUT (output file).
addpath(fullfile(pwd, 'funcs'));
addpath(fullfile(pwd, 'shadow_native'));
rand('state', str2double(getenv('GPC_SEED')));
run(getenv('GPC_VARIANT'));
f = fopen(getenv('GPC_OUT'), 'w');
fwrite(f, [numel(BestCost); BestCost(:)], 'double');
fclose(f);
