function driver

E=[1 1 1 1 1 1 1 0 0 0 0 0;
   1 1 1 1 1 1 1 0 0 0 0 0;
   1 1 0 0 0 0 0 0 0 0 0 0;
   1 1 0 0 0 0 0 0 0 0 0 0;
   1 1 0 0 0 0 0 0 0 0 0 0;
   1 1 1 1 1 1 0 0 0 0 0 0;
   1 1 1 1 1 1 0 0 0 0 0 0;
   1 1 0 0 0 0 0 0 0 0 0 0;
   1 1 0 0 0 0 0 0 0 0 0 0;
   1 1 0 0 0 0 0 0 0 0 0 0;
   1 1 1 1 1 1 1 0 0 0 0 0;
   1 1 1 1 1 1 1 0 0 0 0 0];

H=[0 0 1 1 0 0 0 0 1 1 0 0;
   0 0 1 1 0 0 0 0 1 1 0 0;
   0 0 1 1 0 0 0 0 1 1 0 0;
   0 0 1 1 0 0 0 0 1 1 0 0;
   0 0 1 1 0 0 0 0 1 1 0 0;
   0 0 1 1 1 1 1 1 1 1 0 0;
   0 0 1 1 1 1 1 1 1 1 0 0;
   0 0 1 1 0 0 0 0 1 1 0 0;
   0 0 1 1 0 0 0 0 1 1 0 0;
   0 0 1 1 0 0 0 0 1 1 0 0;
   0 0 1 1 0 0 0 0 1 1 0 0;
   0 0 1 1 0 0 0 0 1 1 0 0];
  
T=[0 1 1 1 1 1 1 1 1 1 1 0;
   0 1 1 1 1 1 1 1 1 1 1 0;
   0 0 0 0 0 1 1 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0;
   0 0 0 0 0 1 1 0 0 0 0 0];

zero=[0 0 0 1 1 1 1 1 1 0 0 0;
      0 0 1 1 1 1 1 1 1 1 0 0;
      0 1 1 1 0 0 0 0 1 1 1 0;
      0 1 1 1 0 0 0 0 1 1 1 0;
      0 1 1 1 0 0 0 0 1 1 1 0;
      0 1 1 1 0 0 0 0 1 1 1 0;
      0 1 1 1 0 0 0 0 1 1 1 0;
      0 1 1 1 0 0 0 0 1 1 1 0;
      0 1 1 1 0 0 0 0 1 1 1 0;
      0 1 1 1 0 0 0 0 1 1 1 0;
      0 0 1 1 1 1 1 1 1 1 0 0;
      0 0 0 1 1 1 1 1 1 0 0 0];

M=[1 1 1 0 0 0 0 0 1 1 1 0;
   1 1 1 1 0 0 0 1 1 1 1 0;
   1 1 1 1 1 0 1 1 1 1 1 0;
   1 1 0 1 1 1 1 1 0 1 1 0;
   1 1 0 0 1 1 1 0 0 1 1 0;
   1 1 0 0 0 1 0 0 0 1 1 0;
   1 1 0 0 0 0 0 0 0 1 1 0;
   1 1 0 0 0 0 0 0 0 1 1 0;
   1 1 0 0 0 0 0 0 0 1 1 0;
   1 1 0 0 0 0 0 0 0 1 1 0;
   1 1 0 0 0 0 0 0 0 1 1 0;
   1 1 0 0 0 0 0 0 0 1 1 0];

EBP=ones(size(E));
HBP=ones(size(H));
TBP=ones(size(T));
zeroBP=ones(size(zero));
MBP=ones(size(M));
for i=1:12
   for j=1:12      
      if E(i,j)==0
         EBP(i,j)=-1;
      end
      if H(i,j)==0
         HBP(i,j)=-1;
      end
      if T(i,j)==0
         TBP(i,j)=-1;
      end
      if zero(i,j)==0
         zeroBP(i,j)=-1;
      end
      if M(i,j)==0
         MBP(i,j)=-1;
      end
   end
end

mu = 0.1;
n = 100000;
tol = 0.4;
p = 30;

memoryE = corrmm(EBP, EBP, mu, n, tol);

corrupt_E = corrupt(EBP, p);

recallE = memoryE * corrupt_E;

diffE = EBP - recallE;

thresholdE = threshold_image(recallE);

diff_err_E = avg_abs_diff(EBP, recallE);

pct_mismatch_E = compute_mismatch(thresholdE, EBP);

disp('Difference Error for E: ')
disp(diff_err_E)

disp('Percentage Mismatch for E: ')
disp(pct_mismatch_E)


memoryH = corrmm(HBP, HBP, mu, n, tol);

corrupt_H = corrupt(HBP, p);

recallH = memoryH * corrupt_H;

diffH = HBP - recallH;

thresholdH = threshold_image(recallH);

diff_err_H = avg_abs_diff(HBP, recallH);

pct_mismatch_H = compute_mismatch(thresholdH, HBP);

disp('Difference Error for H: ')
disp(diff_err_H)

disp('Percentage Mismatch for H: ')
disp(pct_mismatch_H)


memoryM = corrmm(MBP, MBP, mu, n, tol);

corrupt_M = corrupt(MBP, p);

recallM = memoryM * corrupt_M;

diffM = MBP - recallM;

thresholdM = threshold_image(recallM);

diff_err_M = avg_abs_diff(MBP, recallM);

pct_mismatch_M = compute_mismatch(thresholdM, MBP);

disp('Difference Error for M: ')
disp(diff_err_M)

disp('Percentage Mismatch for M: ')
disp(pct_mismatch_M)


memoryT = corrmm(TBP, TBP, mu, n, tol);

corrupt_T = corrupt(TBP, p);

recallT = memoryT * corrupt_T;

diffT = TBP - recallT;

thresholdT = threshold_image(recallT);

diff_err_T = avg_abs_diff(TBP, recallT);

pct_mismatch_T = compute_mismatch(thresholdT, TBP);

disp('Difference Error for T: ')
disp(diff_err_T)

disp('Percentage Mismatch for T: ')
disp(pct_mismatch_T)


memoryzero = corrmm(zeroBP, zeroBP, mu, n, tol);

corrupt_zero = corrupt(zeroBP, p);

recallzero = memoryzero * corrupt_zero;

diffzero = zeroBP - recallzero;

thresholdzero = threshold_image(recallzero);

diff_err_zero = avg_abs_diff(zeroBP, recallzero);

pct_mismatch_zero = compute_mismatch(thresholdzero, zeroBP);

disp('Difference Error for zero: ')
disp(diff_err_zero)

disp('Percentage Mismatch for zero: ')
disp(pct_mismatch_zero)

memory_final = memoryE + memoryH + memoryM + memoryT + memoryzero;
imshow(memory_final * corrupt_E)

input_total = [EBP, HBP, MBP, TBP, zeroBP];

test_total = [corrupt_E, corrupt_H, corrupt_M, corrupt_T, corrupt_zero];

recall_total = [recallE, recallH, recallM, recallT, recallzero];

diff_total = [diffE, diffH, diffM, diffT, diffzero];

threshold_total = [thresholdE, thresholdH, thresholdM, thresholdT, thresholdzero];


figure
imagesc(input_total, [min(min(input_total)), max(max(input_total))])
colormap(gray)
colorbar
set(gca,'xtick',[]), set(gca,'ytick',[])
title('Uncorrupted Training Inputs And Desired Outputs')
saveas(gcf,'total_input.jpg')

figure
imagesc(test_total, [min(min(test_total)), max(max(test_total))])
colormap(gray)
colorbar
set(gca,'xtick',[]), set(gca,'ytick',[])
title(['Input Images with ', num2str(p), '% Corruption for Recall'])
saveas(gcf,'total_test.jpg')

figure
imagesc(recall_total, [min(min(recall_total)), max(max(recall_total))])
colormap(gray)
colorbar
set(gca,'xtick',[]), set(gca,'ytick',[])
title('Recall Images')
saveas(gcf,'total_recall.jpg')

figure
imagesc(diff_total, [min(min(diff_total)), max(max(diff_total))])
colormap(gray)
colorbar
set(gca,'xtick',[]), set(gca,'ytick',[])
title('Difference Between Recalled And Desired Output Images')
saveas(gcf,'total_diff.jpg')

figure
imagesc(threshold_total, [min(min(threshold_total)), max(max(threshold_total))])
colormap(gray)
colorbar
set(gca,'xtick',[]), set(gca,'ytick',[])
title('Thresholded Recalled Images')
saveas(gcf,'total_threshold.jpg')

end


function M=corrmm(X,Y,mu,n,tol)
[nx,mx]=size(X);
[ny,my]=size(Y);
%initialize the memory matrix
M=zeros(ny,nx);
for k=1:n %n is the maximum number of outer loop iterations
    % randomize the ordering of the vector in both X & Y
    RN=randperm(mx);
    X=X(:,[RN]);
    Y=Y(:,[RN]);
    for i=1:mx
        M=M+mu*(Y(:,i)-M*X(:,i))*X(:,i)';
    end
    if norm(Y-M*X)<=tol
        disp('Gradient Search Terminated ===>>> ||Y-M*X||<=tol')
        disp('Number of Iterations = '), disp(k*i)
        break
    end
end
end


function Y = corrupt(X, p)
[num_row, num_col] = size(X);
num_total_pixels = num_row * num_col;

num_corrupt_pixels = ceil(p/100 * num_total_pixels);
x_indices = zeros(1, num_corrupt_pixels);
y_indices = zeros(1, num_corrupt_pixels);

Y = X;

for i = 1:num_corrupt_pixels
    rand_x = ceil(rand() * num_col);
    rand_y = ceil(rand() * num_row);
    
    while includes(x_indices, y_indices, rand_x, rand_y) == true
        rand_x = ceil(rand() * num_col);
        rand_y = ceil(rand() * num_row);
    end
    
    x_indices(i) = rand_x;
    y_indices(i) = rand_y;
        
    if Y(rand_y, rand_x) == 1
        Y(rand_y, rand_x) = -1;
    elseif Y(rand_y, rand_x) == -1
        Y(rand_y, rand_x) = 1;
    end
end

end


function bool = includes(X, Y, x, y)
bool = false;
for i = 1:length(X)
    if x == X(i) && y == Y(i)
        bool = true;
        break
    end
end
end


function thresholded = threshold_image(X)
[n,m] = size(X);
thresholded = zeros(size(X));
for i = 1:n
    for j = 1:m
        if X(i,j) > 0
            thresholded(i,j) = 1;
        elseif X(i,j) < 0
            thresholded(i,j) = -1;
        end
    end
end
end


function err = avg_abs_diff(X,Y)
[n,m] = size(X);
err = 0;
for i = 1:n
    for j = 1:m
        err = err + abs(X(i,j) - Y(i,j));
    end
end
err = err / (n*m);
end


function pct_mismatch = compute_mismatch(X,Y)
[n,m] = size(X);
num_mismatch = 0;
for i = 1:n
    for j = 1:m
        if X(i,j) ~= Y(i,j)
            num_mismatch = num_mismatch + 1;
        end
    end
end
pct_mismatch = num_mismatch / (n*m);
end


