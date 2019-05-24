function [H,U]= ekf_update_mat(model,mu,k,i)
% ACTUALLY, mu here is a 1d column vector. 
    P = model.E2B*mu(1:3,:);
    P1 = model.E2B*model.BM(k,27:29)';
    P2 = model.E2B*model.BM2(k,27:29)';
    dP1 = P-P1;
    dP2 = P-P2;
    dP0 = cat(3,dP1,dP2);
    
    dP = dP0(:,:,i);
    
    H = zeros(model.z_dim,model.x_dim);
    % ASSUMES the relative position of two observers is always [0; 500]
    mag = mu(1,:).*mu(1,:) + mu(2,:).*mu(2,:);
    mag2 = mu(1,:).*mu(1,:) + (mu(2,:)-500).*(mu(2,:)-500);
    
    H(1,1) = -mu(2,:)./mag;
    H(1,2) = mu(1,:)./mag;
    H(2,1) = -(mu(2,:)-500)./mag2;
    H(2,2) = mu(1,:)./mag2;
%    H(1,1) = -dP(1,:)*dP(2,:)/(norm(dP)^2*norm(dP([1 3],:)))*(-1);
%    H(1,2) = norm(dP([1 3],:))/norm(dP)^2;
%    H(1,3) = -dP(3,:)*dP(2,:)/(norm(dP)^2*norm(dP([1 3],:)))*(-1);
%    H(2,1) = dP(3,:)/norm(dP([1 3],:))^2*(-1);
%    H(2,3) = -dP(1,:)/norm(dP([1 3],:))^2*(-1);
    
%     H(1,1) = -dP1(1,:)*dP1(2,:)/(norm(dP1)^2*norm(dP1([1 3],:)))*(-1);
%     H(1,2) = norm(dP1([1 3],:))/norm(dP1)^2;
%     H(1,3) = -dP1(3,:)*dP1(2,:)/(norm(dP1)^2*norm(dP1([1 3],:)))*(-1);
%     H(2,1) = dP1(3,:)/norm(dP1([1 3],:))^2*(-1);
%     H(2,3) = -dP1(1,:)/norm(dP1([1 3],:))^2*(-1);
    
%     H(3,1) = -dP2(1,:)*dP2(2,:)/(norm(dP2)^2*norm(dP2([1 3],:)))*(-1);
%     H(3,2) = norm(dP2([1 3],:))/norm(dP2)^2;
%     H(3,3) = -dP2(3,:)*dP2(2,:)/(norm(dP2)^2*norm(dP2([1 3],:)))*(-1);
%     H(4,1) = dP2(3,:)/norm(dP2([1 3],:))^2*(-1);
%     H(4,3) = -dP2(1,:)/norm(dP2([1 3],:))^2*(-1);   
    
    U = eye(size(model.R));