function z = Michalewicz(x)
m = 10; i = 1:numel(x);
z = -sum(sin(x) .* sin(i .* x.^2 / pi).^(2*m));
end
