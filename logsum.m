function output = logsum(input)
% Logsum performs the calculation of the logrithm of a summation of
% elements. output = log(sum over i (e^input(i)))
% input should be a row vector. If input is a matrix, default take row sum

maxinput = max(input,[],2);
input = bsxfun(@minus,input,maxinput);
input = exp(input);
output = bsxfun(@plus,log(sum(input,2)),maxinput);