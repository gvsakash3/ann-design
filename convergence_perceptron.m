clear all
clc
t=[0;1;0;1];
p1=[2;2]; p2=[1;-2]; p3=[-2;2]; p4=[-1;1];
p=[p1 p2 p3 p4]
W(:,1)=[0;0];%let initial values be null for weight and bias
b(1)=0
w = W;
bias = b;
count = 0;
count1 = 0;
j = 1;
for i=1:50
    a(j)=hardlim(w'*p(:,j)+bias);
    if a(j)==t(j)
        W(:,i+1)=W(:,i);
        b(i+1)=b(i);
    else
        e=t(j)-a(j);
        W(:,i+1)=W(:,i)+e*(p(:,j));
        b(i+1)=b(i)+e;
    end
    w = W(:,i+1);
    bias = b(i+1);
    count = count + 1;
    j = j + 1;
    if count == length(p)
        count = 0;
        for k = 1:length(p)
            if a(k) == t(k);
                count1 = count1 + 1;
            end
            j = 1;
        end
    end
    if count1 == length(p)
        break
    else
         count1 = 0;
    end
end
% for i=1:10
%     for i=1:4
%         a(i)=hardlim(W(:,i)'*p(:,i)+b(i))
%         a(i)=hardlim(W(:,i)'*p(:,i)+b(i))
%         if a(i)==t(i)
%             a=hardlim(W(:,i)'*p(:,i+1)+b(i))
%             W(:,i+1)=W(:,i);
%             b(i+1)=b(i);
%         else
%             e=t(i)-a(i);
%             W(:,i+1)=W(:,i)+e*(p(:,i));
%             b(i+1)=b(i)+e;
%             a=hardlim(W(i+1)*p(:,i+1)+b(i+1))
%         end
%         w = W(:,i+1);
%     end
% end

% for i=1:50
%     a(i)=hardlim(w'*p(:,j)+bias);
%     if a(i)==t(j)
%         W(:,i+1)=W(:,i);
%         b(i+1)=b(i);
%     else
%         e=t(j)-a(i);
%         W(:,i+1)=W(:,i)+e*(p(:,j));
%         b(i+1)=b(i)+e;
%     end
%     w = W(:,i+1);
%     bias = b(i+1);
%     count = count + 1;
%     j = j + 1;
%     if count == length(p)
%         j = 1;
%         count = 0;
%     end
%     if a(i) == t(j)
%         count1 = count1 + 1;
%     else
%         count1 = 1;
%     end
%     if count1 == length(p)
%         break
%     end
% end