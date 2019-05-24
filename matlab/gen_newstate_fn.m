function [X,F,G]= gen_newstate_fn(model,Xd,V)

%nonlinear state space equation (CT model)

if ~isnumeric(V)
    if strcmp(V,'noise')
        V= model.B*randn(size(model.B,2),size(Xd,2));
    elseif strcmp(V,'noiseless')
        V= zeros(size(model.B,1),size(Xd,2));
    end
end

if isempty(Xd)
    X= [];
else %modify below here for user specified transition model
    X= zeros(size(Xd));

dt = model.T;    
ldx  = 0.05;
ldy  = 0.05;
ldz  = 0.05;
A13 = [1/ldx^2*(exp(-ldx*dt)+ldx*dt-1)  0                                0
       0                                1/ldy^2*(exp(-ldy*dt)+ldy*dt-1)  0
       0                                0                                1/ldz^2*(exp(-ldz*dt)+ldz*dt-1)];
   
A23 = [1/ldx*(1-exp(-ldx*dt))           0                                0
       0                                1/ldy*(1-exp(-ldy*dt))           0
       0                                0                                1/ldz*(1-exp(-ldz*dt))];
   
A33 = [exp(-ldx*dt)           0                       0
       0                       exp(-ldy*dt)           0
       0                       0                       exp(-ldz*dt)];

Ak  = [eye(3)   dt*eye(3) A13
       zeros(3) eye(3)    A23
       zeros(3) zeros(3)  A33];
   
X = Ak*Xd;
    %-- add scaled noise 
    X= X+ V;
    
    F = Ak;
    G = eye(size(model.B));
end