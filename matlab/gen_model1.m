function model= gen_model

load siml_res_BM
load siml_res_BM2

model.BM = siml_res_BM;
model.BM2 = siml_res_BM2;
model.E2B = [-1 0 0
              0 1 0
              0 0 -1];
% basic parameters
model.x_dim= 9;   %dimension of state vector
model.z_dim= 2;%4;   %dimension of observation vector
model.v_dim= 9;   %dimension of process noise
model.w_dim= 2;%4;   %dimension of observation noise

% dynamical model parameters (SINGER model)
% state transformation given by gen_newstate_fn, transition matrix is N/A in non-linear case
model.T= 0.02;                         %sampling period
model.B= [0e-2*eye(3)    zeros(3)      zeros(3)
          zeros(3)    0e-2*eye(3)      zeros(3)
          zeros(3)    zeros(3)      1*0.02*eye(3)];
model.Q= model.B*model.B';

% survival/death parameters
model.P_S= .99;
model.Q_S= 1-model.P_S;

% birth parameters (Poisson birth model, multiple Gaussian components)
model.L_birth= 4;                                                     %no. of Gaussian birth terms
model.w_birth= zeros(model.L_birth,1);                                %weights of Gaussian birth terms (per scan) [sum gives average rate of target birth per scan]
model.m_birth= zeros(model.x_dim,model.L_birth);                      %means of Gaussian birth terms 
model.B_birth= zeros(model.x_dim,model.x_dim,model.L_birth);          %std of Gaussian birth terms
model.P_birth= zeros(model.x_dim,model.x_dim,model.L_birth);          %cov of Gaussian birth terms

g = 9.8;%%%%%%%%%%%%

% pos1 = [225e3 400e3 0.25e3]';
% pos2 = [200e3 375e3 0e3]';
% pos3 = [175e3 350e3 -0.25e3]';
% pos4 = [150e3 325e3 -0.5e3]';

pos1 = [225e3 400e3 0.5e3]';
pos2 = [200e3 375e3 0.25e3]';
pos3 = [175e3 350e3 0e3]';
pos4 = [150e3 325e3 -0.25e3]';

pos1 = [225e3 400e3 0.5e3]';
pos2 = [200e3 375e3 0.25e3]';
pos3 = [175e3 350e3 -0.25e3]';
pos4 = [150e3 325e3 -0.5e3]';

% pos1 = [300e3 400e3 1e3]';
% pos2 = [200e3 350e3 0.5e3]';
% pos3 = [100e3 300e3 -0.5e3]';
% pos4 = [0e3 250e3 -1e3]';

vel1 = 0*[3.5e3 -400 10]';
vel2 = 0*[3e3 -350 5]';
vel3 = 0*[2.5e3 -300 -5]';
vel4 = 0*[2e3 -250 -10]';


model.w_birth(1)= 2/100;                                              %birth term 1
model.B_birth(:,:,1)= 1e2*diag([ 500; 500; 500; 50; 50; 50; 50; 50; 50 ]);
model.B_birth(:,:,1)= 1e2*diag([ 500; 500; 50; 50; 5; 5; 0.1; 0.1; 0.1 ]);
% model.B_birth(:,:,1)= 1e2*diag([ 50; 50; 50; 5; 5; 1; 1; 1; 1 ]);
% model.B_birth(:,:,1)= 1e2*diag([ 5; 5; 5; 1; 1; 1; 0.1; 0.1; 0.1 ]);
model.m_birth(:,1)= [ pos1; vel1; zeros(3,1)];
model.P_birth(:,:,1)= model.B_birth(:,:,1)*model.B_birth(:,:,1)';
    
model.w_birth(2)= 3/100;                                              %birth term 2
model.B_birth(:,:,2)= 1e2*diag([ 500; 500; 500; 50; 50; 50; 50; 50; 50 ]);
model.B_birth(:,:,2)= 1e2*diag([ 500; 500; 50; 50; 5; 5; 0.1; 0.1; 0.1 ]);
% model.B_birth(:,:,2)= 1e2*diag([ 500; 500; 500; 50; 50; 5; 5; 5; 5 ]);
% model.B_birth(:,:,2)= 1e2*diag([ 50; 50; 50; 5; 5; 1; 1; 1; 1 ]);
% model.B_birth(:,:,2)= 1e2*diag([ 5; 5; 5; 1; 1; 1; 0.1; 0.1; 0.1 ]);
model.m_birth(:,2)= [ pos2; vel2; zeros(3,1) ];
model.P_birth(:,:,2)= model.B_birth(:,:,2)*model.B_birth(:,:,2)';

model.w_birth(3)= 3/100;                                              %birth term 3
model.B_birth(:,:,3)= 1e2*diag([ 500; 500; 500; 50; 50; 50; 50; 50; 50 ]);
model.B_birth(:,:,3)= 1e2*diag([ 500; 500; 50; 50; 5; 5; 0.1; 0.1; 0.1 ]);
% model.B_birth(:,:,3)= 1e2*diag([ 500; 500; 500; 50; 50; 5; 5; 5; 5 ]);
% model.B_birth(:,:,3)= 1e2*diag([ 50; 50; 50; 5; 5; 1; 1; 1; 1 ]);
% model.B_birth(:,:,3)= 1e2*diag([ 50; 50; 50; 1; 1; 1; 0.1; 0.1; 0.1 ]);
model.m_birth(:,3)= [ pos3; vel3; zeros(3,1) ];
model.P_birth(:,:,3)= model.B_birth(:,:,3)*model.B_birth(:,:,3)';

model.w_birth(4)= 2/100;                                              %birth term 4
model.B_birth(:,:,4)= 1e2*diag([ 500; 500; 500; 50; 50; 50; 50; 50; 50 ]);
model.B_birth(:,:,4)= 1e2*diag([ 500; 500; 50; 50; 5; 5; 0.1; 0.1; 0.1 ]);
% model.B_birth(:,:,4)= 1e2*diag([ 500; 500; 500; 50; 50; 5; 5; 5; 5 ]);
% model.B_birth(:,:,4)= 1e2*diag([ 50; 50; 50; 5; 5; 1; 1; 1; 1 ]);
% model.B_birth(:,:,4)= 1e2*diag([ 50; 50; 50; 1; 1; 1; 0.1; 0.1; 0.1 ]);
model.m_birth(:,4)= [ pos4; vel4; zeros(3,1) ];
model.P_birth(:,:,4)= model.B_birth(:,:,4)*model.B_birth(:,:,4)';

% observation model parameters (noisy qe/qb only)
% measurement transformation given by gen_observation_fn, observation matrix is N/A in non-linear case
model.D= 1*diag([0.02*(pi/180); 0.02*(pi/180)]);%0.02*(pi/180); 0.02*(pi/180)]);      %std for angle and range noise
model.R= model.D*model.D';              %covariance for observation noise

% detection parameters
model.P_D= 1;%.98;   %probability of detection in measurements
model.Q_D= 1-model.P_D; %probability of missed detection in measurements

% clutter parameters
model.lambda_c= 0;%20;                             %poisson average rate of uniform clutter (per scan)
model.range_c= [ -pi/2 pi/2; -pi/2 pi/2 ];          %uniform clutter on r/theta
model.pdf_c= 1/prod(model.range_c(:,2)-model.range_c(:,1)); %uniform clutter density




