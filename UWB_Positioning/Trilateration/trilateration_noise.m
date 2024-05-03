% 앵커 좌표와 태그 좌표
anchorLoc = [0 100; 100 0; 100 100];
c = 3e8; % 전파 속도 (미터/초)

% loc_tag.csv 파일에서 태그 좌표 읽기 (첫 줄 헤더를 건너뜀)
tagLoc = readmatrix('loc_tag.csv');

% noise_TOA.csv 파일에서 ToA 값 읽기 (첫 줄 헤더를 건너뜀)
noise_TOA = readmatrix('noise_TOA.csv');

% noise_TOA에서 앵커 2, 3, 4에 대한 TOA 값만 사용
% noise_TOA_a: noise_TOA의 2, 3, 4번째 열 선택
noise_TOA_a = noise_TOA(:, 2:4);

% 각 태그마다 삼변측량 알고리즘 수행 및 제곱 오차 계산
squared_errors = zeros(size(tagLoc, 1), 1);

for i = 1:size(tagLoc, 1)
    % 앵커와 태그 간의 거리 계산
    % TOA에서 거리를 계산 (거리 = TOA * 전파 속도)
    actualDistances = noise_TOA_a(i, :) * c;

    % 삼변측량 알고리즘 수행
    A = 2 * [anchorLoc(2,1) - anchorLoc(1,1), anchorLoc(2,2) - anchorLoc(1,2);
             anchorLoc(3,1) - anchorLoc(1,1), anchorLoc(3,2) - anchorLoc(1,2)];
    b = [actualDistances(1)^2 - actualDistances(2)^2 + anchorLoc(2,1)^2 - anchorLoc(1,1)^2 + anchorLoc(2,2)^2 - anchorLoc(1,2)^2;
         actualDistances(1)^2 - actualDistances(3)^2 + anchorLoc(3,1)^2 - anchorLoc(1,1)^2 + anchorLoc(3,2)^2 - anchorLoc(1,2)^2];
    est_loc = A\b;

    % 제곱 오차 계산
    squared_errors(i) = norm(tagLoc(i,:) - est_loc')^2;
end

% 제곱 오차의 평균을 이용하여 MSE 계산
MSE = mean(squared_errors);

% 결과 출력 (MSE)
disp(['MSE: ', num2str(MSE)]);
