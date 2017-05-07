function [MatrixProfile, MPindex] = StompSelfJoinGPU(A, SubsequenceLength)
% Compute the self similarity join of time series A
% Usage:
% [matrixProfile, matrixProfileIndex] = StompSelfJoin(A, subLen)
% Output:
%     matrixProfile: matrix porfile of the self-join (vector)
%     matrixProfileIndex: matrix porfile index of the self-join (vector)
% Input:
%     A: input time series (vector)
%     subLen: interested subsequence length (scalar)
% Please find the detailed description of this algorithm in the following
% paper:
% Yan Zhu, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia M. Yeh,
% Gareth Funning, Abdullah Mueen, Philip Brisk, and Eamonn Keogh, "Matrix
% Profile II: Exploiting a Novel Algorithm and GPUs to Break the One Hundred Million Barrier for Time Series Motifs and Joins," IEEE International Conference on Data Mining (ICDM), 2016.

%% set trivial match exclusion zone
exclusionZone = round(SubsequenceLength/4);

%% check input
if SubsequenceLength > length(A)/2
    error('Error: Time series is too short relative to desired subsequence length');
end
if SubsequenceLength < 4
    error('Error: Subsequence length must be at least 4');
end
if length(A) == size(A, 2)
   A = A'; 
end

%% initialization
MatrixProfileLength = length(A) - SubsequenceLength + 1;
MatrixProfile = zeros(MatrixProfileLength, 1);
MPindex = zeros(MatrixProfileLength, 1);
[X, n, sumx2, sumx, meanx, sigmax2, sigmax] = ...
    fastfindNNPre(A, SubsequenceLength);

%% compute the matrix profile
dropval=0;
distanceProfile=zeros(MatrixProfileLength,1);
lastz=zeros(MatrixProfileLength,1);
updatePos=false(MatrixProfileLength,1);

% i=1
subsequence = A(1:1+SubsequenceLength-1);
[distanceProfile(:,1) lastz dropval lastsumy lastsumy2]= fastfindNN(X, subsequence, n, SubsequenceLength, ...
            sumx2, sumx, meanx, sigmax2, sigmax);
distanceProfile(:,1) = abs(distanceProfile);
firstz=lastz;

% apply exclusion zone
exclusionZoneStart = 1;
exclusionZoneEnd = exclusionZone;
distanceProfile(exclusionZoneStart:exclusionZoneEnd) = inf;
        

% evaluate initial matrix profile
MatrixProfile(:) = distanceProfile;
%MPindex(:) = 1;
[dmin, idx] = min(distanceProfile);



ProfileAndIndex = zeros(1,length(MatrixProfile),'uint64');
fmin = typecast(single(dmin),'uint32');

item = typecast([fmin idx], 'uint64');

ProfileAndIndex(1) = item;

for i = 2:length(MatrixProfile)
    item = typecast(single(MatrixProfile(i)),'uint32');
    ProfileAndIndex(i) = typecast([item, 1], 'uint64');
end

k = parallel.gpu.CUDAKernel('STOMP.ptx', 'STOMP.cu', 'WavefrontUpdateSelfJoin');
k.ThreadBlockSize = [1024 1 1];
k.GridSize = [ceil(length(MatrixProfile)/1024) 1 1];

profile = gpuArray(ProfileAndIndex);
MatrixProfile(:) = 0;
MPindex(:) = 0;
QT = gpuArray(lastz);
Ta = gpuArray(A);
Tb = gpuArray(A);
means =gpuArray(meanx);
stds = gpuArray(sigmax);
t = tic();

result = feval(k, QT, Ta, Tb, means, stds, profile, SubsequenceLength, length(MatrixProfile), 0, length(MatrixProfile), 1);
ProfileAndIndex = gather(result);
gpuKernelTime = toc(t)
for i = 1:length(MatrixProfile)
    item = typecast(ProfileAndIndex(i), 'uint64');
    itemMP = typecast(item, 'single');
    itemIdx = typecast(item, 'uint32');
    MPindex(i) = itemIdx(2);
    if i >1
        MPindex(i) = MPindex(i) + 1;
    end
    MatrixProfile(i) = itemMP(1);
    %MatrixProfile(i) = typecast([MatrixProfile(i), 1], 'uint64');
end


% m is winSize
function [X, n, sumx2, sumx, meanx, sigmax2, sigmax] = fastfindNNPre(x, m)
n = length(x);
x(n+1:2*n) = 0;
X = fft(x);
cum_sumx = cumsum(x);
cum_sumx2 =  cumsum(x.^2);
sumx2 = cum_sumx2(m:n)-[0;cum_sumx2(1:n-m)];
sumx = cum_sumx(m:n)-[0;cum_sumx(1:n-m)];
meanx = sumx./m;
sigmax2 = (sumx2./m)-(meanx.^2);
sigmax = sqrt(sigmax2);

% m is winSieze
function [dist lastz dropval sumy sumy2] = fastfindNN(X, y, n, m, sumx2, sumx, meanx, sigmax2, sigmax)
%x is the data, y is the query
%y = (y-mean(y))./std(y,1);                      %Normalize the query
dropval=y(1);
y = y(end:-1:1);                                %Reverse the query
y(m+1:2*n) = 0;

%The main trick of getting dot products in O(n log n) time
Y = fft(y);
Z = X.*Y;
z = ifft(Z);

%compute y stats -- O(n)
sumy = sum(y);
sumy2 = sum(y.^2);
meany=sumy/m;
sigmay2 = sumy2/m-meany^2;
sigmay = sqrt(sigmay2);

%computing the distances -- O(n) time
%dist = (sumx2 - 2*sumx.*meanx + m*(meanx.^2))./sigmax2 - 2*(z(m:n) - sumy.*meanx)./sigmax + sumy2;
%dist = 1-dist./(2*m);

dist = 2*(m-(z(m:n)-m*meanx*meany)./(sigmax*sigmay));
dist = sqrt(dist);
lastz=real(z(m:n));