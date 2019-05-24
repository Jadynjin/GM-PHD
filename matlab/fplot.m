% clear;
% clc;
% close all;

load truth
load est

N = truth.K;
X1 = zeros(N,6);
X2 = zeros(N,6);
X1_est = zeros(N,6);
X2_est = zeros(N,6);
for i = 1:N
    X1(i,:) = truth.X{i}(:,1)';
    X2(i,:) = truth.X{i}(:,2)';
    if size(est.X{i},2) < 2
        if i == 1
            X1_est(i,:) = zeros(1,6);
            X2_est(i,:) = zeros(1,6);   
        else
            X1_est(i,:) = X1_est(i-1,:);
            X2_est(i,:) = X2_est(i-1,:);
        end
    else
        X1_est(i,:) = est.X{i}(:,1)';
        X2_est(i,:) = est.X{i}(:,2)';
    end
end

dX1 = X1-X1_est;
dX2 = X2-X2_est;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = X1;
% X1 = X2;
% X2 = X;

t = 0.02*(1:N);
figure(1);
plot(t,X1(:,1),'r--');
hold on;
plot(t,X1_est(:,1),'b-');
xlabel('t/s');ylabel('x/m');
legend('truth','estimate');

figure(2);
plot(t,X1(:,2),'r--');
hold on;
plot(t,X1_est(:,2),'b-');
xlabel('t/s');ylabel('y/m');
legend('truth','estimate');

figure(3);
plot(t,X1(:,3),'r--');
hold on;
plot(t,X1_est(:,3),'b-');
xlabel('t/s');ylabel('z/m');
legend('truth','estimate');

figure(4);
plot(t,X1(:,4),'r--');
hold on;
plot(t,X1_est(:,4),'b-');
xlabel('t/s');ylabel('V_x/(m/s)');
legend('truth','estimate');

figure(5);
plot(t,X1(:,5),'r--');
hold on;
plot(t,X1_est(:,5),'b-');
xlabel('t/s');ylabel('V_y/(m/s)');
legend('truth','estimate');

figure(6);
plot(t,X1(:,6),'r--');
hold on;
plot(t,X1_est(:,6),'b-');
xlabel('t/s');ylabel('V_z/(m/s)');
legend('truth','estimate');

figure(7);
plot(t,X1(:,7),'r--');
hold on;
plot(t,X1_est(:,7),'b-');
xlabel('t/s');ylabel('a_x/(m^2/s)');
legend('truth','estimate');

figure(8);
plot(t,X1(:,8),'r--');
hold on;
plot(t,X1_est(:,8),'b-');
xlabel('t/s');ylabel('a_y/(m^2/s)');
legend('truth','estimate');

figure(9);
plot(t,X1(:,9),'r--');
hold on;
plot(t,X1_est(:,9),'b-');
xlabel('t/s');ylabel('a_z/(m/s)');
legend('truth','estimate');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(11);
plot(t,X2(:,1),'r--');
hold on;
plot(t,X2_est(:,1),'b-');
xlabel('t/s');ylabel('x/m');
legend('truth','estimate');

figure(12);
plot(t,X2(:,2),'r--');
hold on;
plot(t,X2_est(:,2),'b-');
xlabel('t/s');ylabel('y/m');
legend('truth','estimate');

figure(13);
plot(t,X2(:,3),'r--');
hold on;
plot(t,X2_est(:,3),'b-');
xlabel('t/s');ylabel('z/m');
legend('truth','estimate');

figure(14);
plot(t,X2(:,4),'r--');
hold on;
plot(t,X2_est(:,4),'b-');
xlabel('t/s');ylabel('V_x/(m/s)');
legend('truth','estimate');

figure(15);
plot(t,X2(:,5),'r--');
hold on;
plot(t,X2_est(:,5),'b-');
xlabel('t/s');ylabel('V_y/(m/s)');
legend('truth','estimate');

figure(16);
plot(t,X2(:,6),'r--');
hold on;
plot(t,X2_est(:,6),'b-');
xlabel('t/s');ylabel('V_z/(m/s)');
legend('truth','estimate');

figure(17);
plot(t,X2(:,7),'r--');
hold on;
plot(t,X2_est(:,7),'b-');
xlabel('t/s');ylabel('a_x/(m^2/s)');
legend('truth','estimate');

figure(18);
plot(t,X2(:,8),'r--');
hold on;
plot(t,X2_est(:,8),'b-');
xlabel('t/s');ylabel('a_y/(m^2/s)');
legend('truth','estimate');

figure(19);
plot(t,X2(:,9),'r--');
hold on;
plot(t,X2_est(:,9),'b-');
xlabel('t/s');ylabel('a_z/(m/s)');
legend('truth','estimate');


