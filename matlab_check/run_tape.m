% Runs a variant of GPC.m with the random numbers replayed from tapes.
% Environment: GPC_VARIANT (script file), GPC_TAPE_DIR (tapes), GPC_OUT (output file).
addpath(fullfile(pwd, 'funcs'));
addpath(fullfile(pwd, 'shadow_tape'));
run(getenv('GPC_VARIANT'));
P = reshape([pop.Position], numel(pop(1).Position), [])';
C = [pop.Cost];
f = fopen(getenv('GPC_OUT'), 'w');
fwrite(f, [numel(BestCost); BestCost(:); size(P, 1); size(P, 2); reshape(P', [], 1); C(:)], 'double');
fclose(f);
