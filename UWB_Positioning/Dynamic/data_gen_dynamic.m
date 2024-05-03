% UWB 매개변수 설정
numAnchors = 4;
numTags = 1;

% 앵커 좌표 고정 (2차원)
anchorLoc = [0, 0;
             0, 100;
             100, 0;
             100, 100];

% 데이터 초기화
data_points = 10; % 각 샘플의 데이터 포인트 개수
num_samples = 10000; % 생성할 샘플의 개수
total_data_points = data_points * num_samples;
data = zeros(total_data_points, numAnchors);  % 4개 앵커에서 나오는 TOA 값
loc = zeros(total_data_points, 2);  % 태그의 위치를 저장하기 위한 배열
directions = zeros(total_data_points-1, 2);  % 각 데이터 포인트에서의 방향을 저장하기 위한 배열

for s = 1:num_samples
    % 랜덤한 시작 위치 생성 (2차원)
    start_loc = rand(1, 2) * 100;  % (0,0)에서 (100,100) 사이의 랜덤한 값으로 생성  
    
    % 시작 위치 저장
    loc((s-1)*data_points + 1, :) = start_loc;
    
    % 앵커에서 태그까지의 실제 거리 계산
    actualDistances = sqrt(sum((anchorLoc - start_loc).^2, 2));
    
    % ToA 계산
    ToA = actualDistances / (3*10^8); %physconst('LightSpeed');

    % 데이터 저장
    data((s-1)*data_points + 1, :) = max(abs(ToA), eps);  % TOA 값이 음수가 되지 않도록 수정 (eps는 매우 작은 양수)

    % 속도 설정 (100m/s로 고정)
    velocity = 1.5;

    for i = 2:data_points
        % 시간 간격은 0.1초로 설정
        time_interval = 0.1;

        % 랜덤한 방향 생성 (3차원이므로 z축은 무시)
        direction = randn(1, 3);  % 랜덤한 방향 벡터 생성
        direction = direction / norm(direction);  % 단위 벡터로 정규화
        % 3차원 방향을 2차원으로 변경 (z축 무시)
        direction_2D = direction(1:2);

        % 속도와 방향을 곱하여 이동 거리 계산
        displacement = velocity * direction_2D * time_interval;

        % 새로운 위치 계산
        new_loc = loc((s-1)*data_points + (i-1), :) + displacement;

        % 시작 위치가 (0,0)에서 (100,100) 사이에 있도록 보정
        new_loc = max(min(new_loc, 100), 0);

        % 데이터 저장
        loc((s-1)*data_points + i, :) = new_loc;

        % 앵커에서 태그까지의 실제 거리 계산
        actualDistances = sqrt(sum((anchorLoc - new_loc).^2, 2));

        % ToA 계산
        ToA = actualDistances / (3*10^8); %physconst('LightSpeed');

        % 데이터 저장
        data((s-1)*data_points + i, :) = max(abs(ToA), eps);  % TOA 값이 음수가 되지 않도록 수정 (eps는 매우 작은 양수)
        
        % 방향 저장
        directions((s-1)*data_points + (i-1), :) = direction_2D;
    end
end

% TOA 데이터를 CSV 파일로 저장
csvwrite('TOA_data.csv', data);

% 좌표값 데이터를 CSV 파일로 저장
csvwrite('location_data.csv', loc);

% 2개의 샘플 선택
num_samples_to_plot = 2;
sample_indices = randperm(num_samples, num_samples_to_plot);

% 그래프로 그릴 데이터 초기화
sample_TOA_data = cell(num_samples_to_plot, 1);
sample_location_data = cell(num_samples_to_plot, 1);
sample_direction_data = cell(num_samples_to_plot, 1);

% 선택된 샘플의 데이터 추출
for i = 1:num_samples_to_plot
    sample_TOA_data{i} = data((sample_indices(i)-1)*data_points + 1:sample_indices(i)*data_points, :);
    sample_location_data{i} = loc((sample_indices(i)-1)*data_points + 1:sample_indices(i)*data_points, :);
    sample_direction_data{i} = directions((sample_indices(i)-1)*data_points + 1:sample_indices(i)*data_points, :);
end

% 선택된 샘플의 위치와 방향을 그래프로 그리기
figure;
for i = 1:num_samples_to_plot
    subplot(1, num_samples_to_plot, i);
    plot(sample_location_data{i}(:, 1), sample_location_data{i}(:, 2), 'b.', 'MarkerSize', 15);
    hold on;
    quiver(sample_location_data{i}(:, 1), sample_location_data{i}(:, 2), ...
           sample_direction_data{i}(:, 1), sample_direction_data{i}(:, 2), 'r', 'LineWidth', 2);
    % 각 샘플의 위치를 선으로 연결
    plot(sample_location_data{i}(:, 1), sample_location_data{i}(:, 2), 'b-');
    title(['Sample ', num2str(sample_indices(i))]);
    xlabel('X');
    ylabel('Y');
    grid on;
end