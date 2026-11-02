function z = Beale(x)
  x1 = x(1); x2 = x(2);
  z = (1.5 - x1 + x1*x2)^2 + (2.25 - x1 + x1*x2^2)^2 + (2.625 - x1 + x1*x2^3)^2;
end
