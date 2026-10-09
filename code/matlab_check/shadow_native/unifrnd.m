function r = unifrnd(a, b, varargin)
% Same algorithm as unifrnd of the MATLAB Statistics Toolbox:
% r = (a+b)/2 + (b-a)/2 .* (2*rand(sz)-1)   (avoids overflow of a+(b-a)*rand)
sz = [varargin{:}];
a2 = a/2; b2 = b/2; mu = a2 + b2; sig = b2 - a2;
r = mu + sig .* (2*rand(sz) - 1);
end
