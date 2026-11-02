function r = unifrnd(a, b, sz)
  % Replays uniform numbers U from GPC_TAPE_DIR/unifrnd.bin and returns a + (b - a) .* U.
  persistent T k
  if isempty(T)
    f = fopen(fullfile(getenv('GPC_TAPE_DIR'), 'unifrnd.bin'), 'r'); T = fread(f, Inf, 'double'); fclose(f); k = 0;
  end
  n = prod(sz);
  U = reshape(T(k+1:k+n), sz);
  k = k + n;
  r = a + (b - a) .* U;
end
