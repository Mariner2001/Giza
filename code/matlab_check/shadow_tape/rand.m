function r = rand(varargin)
% Replaces the built-in rand: returns the next numbers of the stream rand.bin (GPC_TAPE_DIR).
% rand('__status__') returns [numbers used, numbers available].
persistent tape pos
if isempty(tape)
  fid = fopen(fullfile(getenv('GPC_TAPE_DIR'), 'rand.bin'), 'r');
  tape = fread(fid, Inf, 'double'); fclose(fid); pos = 0;
end
if nargin == 1 && ischar(varargin{1})
  r = [pos numel(tape)]; return
end
if nargin == 0
  sz = [1 1];
elseif nargin == 1
  sz = varargin{1}; if isscalar(sz), sz = [sz sz]; end
else
  sz = [varargin{:}];
end
m = prod(sz);
if pos + m > numel(tape), error('rand stream exhausted'); end
r = reshape(tape(pos+1:pos+m), sz);
pos = pos + m;
end
