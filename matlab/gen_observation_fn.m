function Z= gen_observation_fn(model,X,W,k,i)

%r/t observation equation

if ~isnumeric(W)
    if strcmp(W,'noise')
        W= model.D*randn(size(model.D,2),size(X,2));
    elseif strcmp(W,'noiseless')
        W= zeros(size(model.D,1),size(X,2));
    end
end

if isempty(X)
    Z= [];
else %modify below here for user specified measurement model
    P = model.E2B*X(1:3,:);
    P1 = model.E2B*model.BM(k,27:29)';
    P2 = model.E2B*model.BM2(k,27:29)';
    dP1 = P-P1;
    dP2 = P-P2;
    dP0 = cat(3,dP1,dP2);
    dP = dP0(:,:,i);
%     Z(1,:)= atan2(dP1(2,:),norm(dP1([1 3],:)));   
%     Z(2,:)= atan2(-dP1(3,:),dP1(1,:));
%     Z(1,:)= atan2(dP(2,:),norm(dP([1 3],:)));   
%     Z(2,:)= atan2(-dP(3,:),dP(1,:));
%     Z(3,:)= atan2(dP2(2,:),norm(dP2([1 3],:)));
%     Z(4,:)= atan2(-dP2(3,:),dP2(1,:));
   
    % ASSUMES the relative position of two observers is always [0; 500]
    Z(1,:)= atan2(X(2,:), X(1,:));   
    Z(2,:)= atan2(X(2,:)-500, X(1,:));
%     Z(3,:)= atan2(dP2(2,:),norm(dP2([1 3],:)));
%     Z(4,:)= atan2(-dP2(3,:),dP2(1,:));
    Z= Z+ W;
end
