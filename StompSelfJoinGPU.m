function [MatrixProfile, MPindex] = StompSelfJoinGPU(A, SubsequenceLength)
% Compute the self similarity join of time series A
% Usage:
% [matrixProfile, matrixProfileIndex] = StompSelfJoinGPU(A, subLen)
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

distanceProfile=zeros(MatrixProfileLength,1);
lastz=zeros(MatrixProfileLength,1);

subsequence = A(1:1+SubsequenceLength-1);
subsequence = subsequence(end:-1:1);                                %Reverse the query
subsequence(SubsequenceLength+1:length(A)) = 0;

Y = fft(subsequence);
Z = X.*Y;
z = ifft(Z);
lastz=real(z(SubsequenceLength:MatrixProfileLength));

ProfileAndIndex = zeros(1,length(MatrixProfile),'uint64');

for i = 1:MatrixProfileLength
    item = typecast(single(-inf),'uint32');
    ProfileAndIndex(i) = typecast([item, 1], 'uint64');
end

%Do not modify the following 3 lines unless you know exactly what you are doing
%the block size of 512 is a template parameter to the compiled kernel if you change this you
%will need to modify and recompile the source
k = parallel.gpu.CUDAKernel('STOMP.ptx', 'STOMP.cu', 'WavefrontUpdateSelfJoin');
k.ThreadBlockSize = [512 1 1];
k.GridSize = [ceil(length(MatrixProfile)/ 512) 1 1];

profile = gpuArray(ProfileAndIndex);
MPindex(:) = 0;
sigmax = 1 ./ sigmax;

QT = gpuArray(lastz);
Ta = gpuArray(A);
means =gpuArray(meanx);
stds = gpuArray(sigmax);
t = tic();

result = feval(k, QT, Ta, stds, means, profile, SubsequenceLength, MatrixProfileLength, 0, 1);
ProfileAndIndex = gather(result);
gpuKernelTime = toc(t)
for i = 1:length(MatrixProfile)
    item = typecast(ProfileAndIndex(i), 'uint64');
    itemMP = typecast(item, 'single');
    itemIdx = typecast(item, 'uint32');
    MPindex(i) = itemIdx(2) + 1;
    MatrixProfile(i) = itemMP(1);
end

MatrixProfile = sqrt(max(2 .* (SubsequenceLength - MatrixProfile), 0));

% m is winSize
function [X, n, sumx2, sumx, meanx, sigmax2, sigmax] = fastfindNNPre(x, m)
n = length(x);
X = fft(x);
cum_sumx = cumsum(x);
cum_sumx2 =  cumsum(x.^2);
sumx2 = cum_sumx2(m:n)-[0;cum_sumx2(1:n-m)];
sumx = cum_sumx(m:n)-[0;cum_sumx(1:n-m)];
meanx = sumx./m;
sigmax2 = (sumx2./m)-(meanx.^2);
sigmax = sqrt(sigmax2);

