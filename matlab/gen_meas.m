function meas= gen_meas(model,truth)

%variables
meas.K= truth.K;
meas.S = 1;
meas.Z= cell(truth.K, 1);

%generate measurements
for k=1:truth.K
    if truth.N(k) > 0
        idx= find( rand(truth.N(k),1) <= model.P_D );                                            %detected target indices
        meas.Z{k}= gen_observation_fn(model,truth.X{k}(:,idx),'noise',k,1);                          %single target observations if detected 
    end
    
%     N_c= poissrnd(model.lambda_c);                                                               %number of clutter points
%     C= repmat(model.range_c(:,1),[1 N_c])+ diag(model.range_c*[ -1; 1 ])*rand(model.z_dim,N_c);  %clutter generation
%     meas.Z{k}= [ meas.Z{k} C ];                                                                  %measurement is union of detections and clutter
end
    
