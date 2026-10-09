function z = Rastrigin(x)
z = 10*numel(x) + sum(x.^2 - 10*cos(2*pi*x));
end
