function confusion = make_confusion_matrix(Y_pred, Y_true)

classes = unique(Y_true);
c = length(classes);
confusion = zeros(c);
N = size(Y_true,1);
if(size(Y_true) ~= size(Y_pred)); error("Inconsistent Matrix Sizes"); end
if(size(Y_true,2) == 1); doit = 1; else; doit = 2; end

if (doit == 1)
    if(min(min(Y_true)) == 0); Y_true = Y_true + 1; Y_pred = Y_pred + 1; end
    for i = 1:N
        inx1 = Y_true(i); inx2 = Y_pred(i);
        confusion(inx1, inx2) = confusion(inx1, inx2) + 1;
    end
else
    for i = 1:N
        inx1 = find(Y_true(i,:) == 1); inx2 = find(Y_pred(i,:) == 1);
        confusion(inx1, inx2) = confusion(inx1, inx2) + 1;
    end
end