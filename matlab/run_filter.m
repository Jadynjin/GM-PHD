function est = run_filter(model,meas)

% This is the MATLAB code for the GM-PHD filter proposed in
% (assuming no target spawning)
% B.-N. Vo, and W. K. Ma, "The Gaussian mixture Probability Hypothesis Density Filter," IEEE Trans Signal Processing, Vol. 54, No. 11, pp. 4091-4104, 2006.
% http://ba-ngu.vo-au.com/vo/VM_GMPHD_SP06.pdf
% ---BibTeX entry
% @ARTICLE{GMPHD,
% author={B.-N. Vo and W.-K. Ma},
% journal={IEEE Transactions on Signal Processing},
% title={The Gaussian Mixture Probability Hypothesis Density Filter},
% year={2006},
% month={Nov},
% volume={54},
% number={11},
% pages={4091-4104}}
%---

%=== Setup

%output variables
est.X= cell(meas.K,1);
est.N= zeros(meas.K,1);
est.L= cell(meas.K,1);

%filter parameters
filter.L_max= 100;                  %limit on number of Gaussians
filter.elim_threshold= 1e-5;        %pruning threshold
filter.merge_threshold= 4;          %merging threshold

filter.P_G= 0.9999999;                           %gate size in percentage
filter.gamma= chi2inv(filter.P_G,model.z_dim);   %inv chi^2 dn gamma value
filter.gate_flag= 0;%1;                             %gating on or off 1/0

filter.run_flag= 'disp';            %'disp' or 'silence' for on the fly output

est.filter= filter;

%=== Filtering 

%initial prior
w_update(1)= eps;
% m_update(:,1) = [200e3; 365e3; 0; 3e3; 360; 0; 0; 0; 0]; %.*randn(9,1);
m_update(:,1)= [ 200e3; 365e3; 0; 10; 100; 0];%.*randn(9,1);
% m_update(:,1)= randn(9,1);
% P_update(:,:,1) = diag([500; 500; 500; 50; 50; 50; 5; 5; 5]).^2;
P_update(:,:,1)= 1e0*diag([ 100e0; 100e0; 10; 10; 10; 10]).^2;
L_update = 1;

%recursive filtering
for k=1:meas.K
    %---prediction
    [m_predict,P_predict] = ekf_predict_multiple(model,m_update,P_update);                          %surviving components
    w_predict= model.P_S*w_update;                                                                  %surviving weights
    m_predict= cat(2,model.m_birth,m_predict); P_predict=cat(3,model.P_birth,P_predict);            %append birth components
    w_predict= cat(1,model.w_birth,w_predict);                                                      %append birth weights
                                                 
    L_predict= model.L_birth+L_update;                                                              %number of predicted components
    
    %---gating
    if filter.gate_flag
        meas.Z{k}= gate_meas_ekf(meas.Z{k},filter.gamma,model,m_predict,P_predict);        
    end
   
    for i = 1:meas.S%%%%%%
        
    %---update
    %number of measurements
    m= size(meas.Z{k,i},2);
    
    %missed detection term 
    w_update = model.Q_D*w_predict;
    m_update = m_predict;
    P_update = P_predict;
    
    if m~=0
        %m detection terms 
        [qz_temp,m_temp,P_temp] = ekf_update_multiple(meas.Z{k},model,m_predict,P_predict,k,i);
        for ell=1:m
            w_temp = model.P_D*w_predict(:).*qz_temp(:,ell);
%             if sum(w_temp) > 0 
                w_temp = w_temp./(model.lambda_c*model.pdf_c + sum(w_temp));
%             else
%                 w_temp = 1/numel(w_temp)*ones(size(w_temp));
%             end
            w_update = cat(1,w_update,w_temp);
            m_update = cat(2,m_update,m_temp(:,:,ell));
            P_update = cat(3,P_update,P_temp);
        end
    end 
    
    w_predict = w_update;
    m_predict = m_update;
    P_predict = P_update;
    
    end%%%%%%
            
    %---mixture management
    L_posterior= length(w_update);
    
    %pruning, merging, capping
    [w_update,m_update,P_update]= gaus_prune(w_update,m_update,P_update,filter.elim_threshold);    L_prune= length(w_update);
    [w_update,m_update,P_update]= gaus_merge(w_update,m_update,P_update,filter.merge_threshold);   L_merge= length(w_update);
    [w_update,m_update,P_update]= gaus_cap(w_update,m_update,P_update,filter.L_max);               L_cap  = length(w_update);
    
    L_update= L_cap;
    
    %--- state extraction
    idx= find(w_update > 0.5 );
    for j=1:length(idx)
        repeat_num_targets= round(w_update(idx(j)));
        est.X{k}= [ est.X{k} repmat(m_update(:,idx(j)),[1,repeat_num_targets]) ];
        est.N(k)= est.N(k)+repeat_num_targets;
        est.L{k}= [];
    end
    
    %---display diagnostics
    if ~strcmp(filter.run_flag,'silence')
        disp([' time= ',num2str(k),...
         ' #est mean=' num2str(sum(w_update),4),...
         ' #est card=' num2str(est.N(k),4),...
         ' #gaus orig=',num2str(L_posterior),...
         ' #gaus elim=',num2str(L_prune), ...
         ' #gaus merg=',num2str(L_merge)   ]);
    end

end

save est
            