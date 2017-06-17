function [Wn] = normalizeWeights(W,N)
%Normalize weights and remove values from outside the [0-1] range
Wn = uint8(W*255);
Wn = double(Wn)/255;
Wn = Wn + 1E-12;
Wn = Wn./repmat(sum(Wn,3),[1 1 N]);
end