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


