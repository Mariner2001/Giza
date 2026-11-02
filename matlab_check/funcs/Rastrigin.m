function z = Rastrigin(x)
  n = numel(x);
  z = 10*n + sum(x.^2 - 10*cos(2*pi*x));
end
