%%삼변측량 코드
% 앵커 좌표와 태그 좌표
anchorLoc = [0 100; 100 0; 100 100];
num_tags = 12000;
tagLoc = rand(num_tags, 2) * 100; % 태그의 랜덤한 예상 좌표 생성 (0부터 100까지)

% 각 태그마다 삼변측량 알고리즘 수행 및 제곱 오차 계산
squared_errors = zeros(num_tags, 1);

for i = 1:num_tags
    % 앵커와 태그 간의 거리 계산
    actualDistances = zeros(3,1);
    for j = 1:3
        % 각 앵커와 태그 간의 거리 계산
        actualDistances(j) = norm(anchorLoc(j,:) - tagLoc(i,:));

        % 실제 거리 범위의 1% 계산
        error_range = actualDistances * 0.01;

        % 무작위 오차 생성 
        random_error = 2 * error_range .* rand(size(actualDistances)) - error_range;

        % 실제 거리에 무작위 오차 추가 또는 빼기
        noise_distances = actualDistances + random_error;
        
    end

    % 삼변측량 알고리즘 수행
    A = 2 * [anchorLoc(2,1) - anchorLoc(1,1), anchorLoc(2,2) - anchorLoc(1,2);
             anchorLoc(3,1) - anchorLoc(1,1), anchorLoc(3,2) - anchorLoc(1,2)];
    b = [noise_distances(1)^2 - noise_distances(2)^2 + anchorLoc(2,1)^2 - anchorLoc(1,1)^2 + anchorLoc(2,2)^2 - anchorLoc(1,2)^2;
         noise_distances(1)^2 - noise_distances(3)^2 + anchorLoc(3,1)^2 - anchorLoc(1,1)^2 + anchorLoc(3,2)^2 - anchorLoc(1,2)^2];
    est_loc = A\b;

    % 제곱 오차 계산
    squared_errors(i) = norm(tagLoc(i,:) - est_loc')^2;
end

% 제곱 오차의 평균을 이용하여 MSE 계산
MSE = mean(squared_errors);

% 결과 출력 (MSE)
disp(['MSE: ', num2str(MSE)]);