function r = unifrnd(a, b, varargin)
% Replaces unifrnd: r = a + (b-a).*u, with u the next numbers of the stream unifrnd.bin (uniform on [0,1)).
persistent tape pos
if isempty(tape)
  fid = fopen(fullfile(getenv('GPC_TAPE_DIR'), 'unifrnd.bin'), 'r');
  tape = fread(fid, Inf, 'double'); fclose(fid); pos = 0;
end
if ischar(a)
  r = [pos numel(tape)]; return
end
sz = [varargin{:}];
m = prod(sz);
if pos + m > numel(tape), error('unifrnd stream exhausted'); end
u = reshape(tape(pos+1:pos+m), sz);
pos = pos + m;
r = a + (b - a) .* u;
end
