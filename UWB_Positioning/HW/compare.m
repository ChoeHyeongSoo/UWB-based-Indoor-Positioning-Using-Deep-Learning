% 데이터 읽기
loc_data = readmatrix('data/raw/location.csv');
dnn_data = readmatrix('data/total_dnn_pred_coor.csv'); 
loc_data = loc_data(1:250,1:2);

% Ground Truth Point
rectangle_points = [...
    0.9663, 4.62724; % 점 1
    0.9663, 1.61536; % 점 2
    3.0635, 1.61536; % 점 3
    3.0635, 4.62724  % 점 4
];

% Ground Truth 생성
rectangle_lines = [...
    rectangle_points(1,:), rectangle_points(2,:);
    rectangle_points(2,:), rectangle_points(3,:);
    rectangle_points(3,:), rectangle_points(4,:);
    rectangle_points(4,:), rectangle_points(1,:);
];

% HW 데이터 최소거리 계산
num_points = size(loc_data, 1);
distances_loc = zeros(num_points, 1);

% 플롯
figure;
hold on;
scatter(loc_data(:, 1), loc_data(:, 2), 'b');

for i = 1:num_points
    point = loc_data(i, :);
    min_distance = inf;
    for j = 1:4
        line = [rectangle_lines(j, 1:2); rectangle_lines(j, 3:4)];
        [distance, ~, ~] = point_to_line_dist(point, line);
        if distance < min_distance
            min_distance = distance;
        end
    end
    distances_loc(i) = min_distance;
    
    % 데이터 포인트와 사각형까지의 최소 거리를 시각화
    nearest_point = point_nearest_line(point, rectangle_lines);
    plot([point(1), nearest_point(1)], [point(2), nearest_point(2)], 'r-');
end

% HW의 MSE 계산
mse_loc = mean(distances_loc .^ 2);

% 결과 출력
fprintf('HW의 MSE: %.6f\n', mse_loc);

% ML 파일의 데이터 포인트와 사각형 변 간의 최소 거리 계산
num_points_dnn = size(dnn_data, 1);
distances_dnn = zeros(num_points_dnn, 1);

% ML 파일의 데이터 포인트 시각화
scatter(dnn_data(:, 1), dnn_data(:, 2), 'g');

for i = 1:num_points_dnn
    point = dnn_data(i, :);
    min_distance = inf;
    for j = 1:4
        line = [rectangle_lines(j, 1:2); rectangle_lines(j, 3:4)];
        [distance, ~, ~] = point_to_line_dist(point, line);
        if distance < min_distance
            min_distance = distance;
        end
    end
    distances_dnn(i) = min_distance;
    
    % 데이터 포인트와 사각형까지의 최소 거리를 시각화
    nearest_point = point_nearest_line(point, rectangle_lines);
    plot([point(1), nearest_point(1)], [point(2), nearest_point(2)], 'y-');
end

% 모델 MSE 계산
mse_dnn = mean(distances_dnn .^ 2);

fprintf('ML의 MSE: %.6f\n', mse_dnn);

% 사각형을 시각화
plot(rectangle_points(:,1), rectangle_points(:,2), 'k-', 'LineWidth', 2);

% 시각화 설정
xlabel('X 좌표');
ylabel('Y 좌표');
title('데이터 포인트와 사각형까지의 최소 거리 시각화');
grid on;
hold off;

% point_to_line_dist 함수 정의
function [min_dist, nearest_x, nearest_y] = point_to_line_dist(point, line)
    % 점 `point`와 선분 `line` 사이의 최소 거리 계산
    x0 = point(1);
    y0 = point(2);
    x1 = line(1,1);
    y1 = line(1,2);
    x2 = line(2,1);
    y2 = line(2,2);
    
    % 선분 길이 계산
    line_length = hypot(x2 - x1, y2 - y1);
    
    % 선분이 점이 아닌 경우
    if line_length == 0
        min_dist = hypot(x0 - x1, y0 - y1);
        nearest_x = x1;
        nearest_y = y1;
    else
        % 파라미터 t를 사용하여 점과 선분 사이의 거리를 계산
        t = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_length^2;
        t = max(0, min(1, t)); % t는 0과 1 사이로 제한
        nearest_x = x1 + t * (x2 - x1);
        nearest_y = y1 + t * (y2 - y1);
        min_dist = hypot(x0 - nearest_x, y0 - nearest_y);
    end
end

% point_nearest_line 함수 정의
function nearest_point = point_nearest_line(point, rectangle_lines)
    min_distance = inf;
    nearest_point = [];
    
    for j = 1:4
        line = [rectangle_lines(j, 1:2); rectangle_lines(j, 3:4)];
        [distance, nearest_x, nearest_y] = point_to_line_dist(point, line);
        if distance < min_distance
        min_distance = distance;
            nearest_point = [nearest_x, nearest_y];
        end
    end
end