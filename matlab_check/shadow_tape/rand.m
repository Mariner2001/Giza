function r = rand(varargin)
  % Replays uniform numbers from GPC_TAPE_DIR/rand.bin (doubles, in the order GPC.m consumes them).
  persistent T k
  if isempty(T)
    f = fopen(fullfile(getenv('GPC_TAPE_DIR'), 'rand.bin'), 'r'); T = fread(f, Inf, 'double'); fclose(f); k = 0;
  end
  if nargin == 0
    sz = [1 1];
  else
    sz = [varargin{:}];
    if numel(sz) == 1, sz = [sz sz]; end
  end
  n = prod(sz);
  r = reshape(T(k+1:k+n), sz);
  k = k + n;
end
