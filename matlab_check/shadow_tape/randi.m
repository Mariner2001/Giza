function r = randi(lim, varargin)
  % Replays integers from GPC_TAPE_DIR/randi.bin.
  persistent T k
  if isempty(T)
    f = fopen(fullfile(getenv('GPC_TAPE_DIR'), 'randi.bin'), 'r'); T = fread(f, Inf, 'double'); fclose(f); k = 0;
  end
  k = k + 1;
  r = T(k);
end
