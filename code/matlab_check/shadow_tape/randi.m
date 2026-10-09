function r = randi(varargin)
% Replaces the built-in randi for the only call in GPC.m, randi([1 n]): returns the next integer of randi.bin.
persistent tape pos
if isempty(tape)
  fid = fopen(fullfile(getenv('GPC_TAPE_DIR'), 'randi.bin'), 'r');
  tape = fread(fid, Inf, 'double'); fclose(fid); pos = 0;
end
if nargin == 1 && ischar(varargin{1})
  r = [pos numel(tape)]; return
end
lim = varargin{1};
if nargin ~= 1 || numel(lim) ~= 2, error('unexpected call to randi'); end
if pos + 1 > numel(tape), error('randi stream exhausted'); end
r = tape(pos+1); pos = pos + 1;
if r < lim(1) || r > lim(2), error('randi value out of range'); end
end
