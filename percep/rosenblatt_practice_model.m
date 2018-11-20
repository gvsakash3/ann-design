clear all;
r=10;
w=6;
d=-1;
rand('state',0);
rho=r+(rand(1000,1)-1/2)*w;
rand('state',4);
theta1=pi*rand(1000,1);
D=zeros(2000,2);
D(1:1000,:)=[rho.*cos(theta1),rho.*sin(theta1)]; %the upper moon

theta2=pi+pi*rand(1000,1);
D(1001:2000,:)=[r+rho.*cos(theta2),d+rho.*sin(theta2)];
m=3;
mu=.1;
itermax=50; %the maximum of iterative
x(1,:)=ones(1,2000);
x(2,:)=D(:,1)';
x(3,:)=D(:,2);
w=zeros(m,1);
goal(1:1000,1)=ones(1000,1); %target
goal(1001:2000,1)=-1*ones(1000,1);
y=ones(2000,1);% output if choose y=-ones(2000,1), this means if w'*x<=0,
% x belongs to the first category
k=1;
while k<=itermax
for i=1:2000
w=w+mu*(goal(i)-y(i))*x(:,i);
%         w=w+x(:,i);
y(i)=sign(w'*x(:,i));
end
error(k)=length(find(y~=goal))/length(goal);
if y==goal
break;
end
k=k+1;
end
figure(1);
plot(D(:,1),D(:,2),'.');
f=@(x) -w(1)/w(3)-w(2)/w(3).*x;
s=-15:0.1:25;
hold on; plot(s,f(s),'k');
figure(2);
plot(error)
