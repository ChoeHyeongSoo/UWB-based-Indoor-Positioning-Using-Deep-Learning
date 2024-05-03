%% 고정좌표 매트랩 코드 
% UWB 매개변수 설정
numAnchors = 4;
numTags = 12000;  % 12000개의 데이터 생성 

% 앵커 좌표 고정
anchorLoc = [0, 0;
             0, 100;
             100, 0;
             100, 100];

% 빛의 속도 설정 (미터/초) 
lightSpeed = 299792458; 

% 데이터 초기화
data_TOA = zeros(numTags, numAnchors);  % 4개 앵커에서 나오는 TOA 값 
loc_tag = zeros(numTags, 2);  % 태그의 위치를 저장하기 위한 배열 

for i = 1:numTags
    % 랜덤한 태그 위치 생성 
    tagLoc = [rand()*100, rand()*100];
    
    % 앵커에서 태그까지의 실제 거리 계산
    actualDistances = sqrt(sum((anchorLoc - tagLoc).^2, 2));

    % 실제 거리 범위의 1% 계산
    error_range = actualDistances * 0.01;

    % 무작위 오차 생성 (-error_range부터 +error_range까지의 값)
    random_error = 2 * error_range .* rand(size(actualDistances)) - error_range;

    % 실제 거리에 무작위 오차 추가 또는 빼기
    noise_distances = actualDistances + random_error;

    % 실제 TOF 계산
    Ideal_TOF = actualDistances / lightSpeed;

    %noise TOA 계산
    Noise_TOA = noise_distances / lightSpeed;

    % 데이터 저장
    data_TOA(i, :) = Ideal_TOF';  
    loc_tag(i,:) = tagLoc';
    noise_TOA(i,:) = Noise_TOA';
    
end

% 데이터 출력 (첫 10개 행만 출력)
disp('첫 10개의 데이터:');
disp(data_TOA(1:10, :));

% 데이터를 CSV 파일로 저장
writematrix(data_TOA, 'data_TOA.csv');
writematrix(loc_tag, 'loc_tag.csv');
writematrix(noise_TOA, 'noise_TOA.csv');