function r = unifrnd(a, b, sz)
  % unifrnd of the MATLAB Statistics Toolbox (not part of core Octave), using Octave's rand.
  r = (a + b) / 2 + (b - a) / 2 .* (2 * rand(sz) - 1);
end
