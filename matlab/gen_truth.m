function truth= gen_truth(model)

tf_detect = 23.5;%%%%
n = tf_detect/model.T;

%variables
truth.K= n;                   %length of data/number of scans
truth.X= cell(truth.K,1);             %ground truth for states of targets  
truth.N= zeros(truth.K,1);            %ground truth for number of targets
truth.L= cell(truth.K,1);             %ground truth for labels of targets (k,i)
truth.track_list= cell(truth.K,1);    %absolute index target identities (plotting)
truth.total_tracks= 0;          %total number of appearing tracks

%target initial states and birth/death times
nbirths= 2;%10;

xstart(:,1)  = [model.BM(1,5:7)'; model.BM(1,2:4)'];      tbirth(1)  = 1;     tdeath(1)  = truth.K+1;
xstart(:,2)  = [model.BM2(1,5:7)'; model.BM2(1,2:4)'];         tbirth(2)  = 1;    tdeath(2)  = truth.K+1;
% xstart(:,3)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(3)  = 10;    tdeath(3)  = truth.K+1;
% xstart(:,4)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(4)  = 10;    tdeath(4)  = 66;
% xstart(:,5)  = [ 250-3.8676; 11; 750-11.0747; 5; wturn/4 ];             tbirth(5)  = 20;    tdeath(5)  = 80;
% xstart(:,6)  = [ -250+7.3806; -12; 1000-6.7993; -12; wturn/2 ];         tbirth(6)  = 40;    tdeath(6)  = truth.K+1;
% xstart(:,7)  = [ 1000; 0; 1500; -10; wturn/4 ];                         tbirth(7)  = 40;    tdeath(7)  = truth.K+1;
% xstart(:,8)  = [ 250; -50; 750; 0; -wturn/4 ];                          tbirth(8)  = 40;    tdeath(8)  = 80;
% xstart(:,9)  = [ 1000; -50; 1500; 0; -wturn/4 ];                        tbirth(9)  = 60;     tdeath(9)  = truth.K+1;
% xstart(:,10)  = [ 250; -40; 750; 25; wturn/4 ];                         tbirth(10)  = 60;    tdeath(10)  = truth.K+1;

%generate the tracks
for targetnum=1:nbirths
    targetstate = xstart(:,targetnum);
    for k=tbirth(targetnum):min(tdeath(targetnum),truth.K)
%         targetstate = gen_newstate_fn(model,targetstate,'noiseless');
if targetnum==1
        % targetstate = [model.BM(k,5:7)'; model.BM(k,2:4)'; model.E2B'*model.BM(k.69:71)'];
        targetstate = [model.BM(k,5:7)'; model.BM(k,2:4)'];
elseif targetnum==2
    targetstate = [model.BM2(k,5:7)'; model.BM2(k,2:4)'];
end
        truth.X{k}= [truth.X{k} targetstate];
        truth.track_list{k} = [truth.track_list{k} targetnum];
        truth.N(k) = truth.N(k) + 1;
     end
end
truth.total_tracks= nbirths;

save truth
